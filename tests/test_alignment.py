"""5-8, 10. Timing calculation, atempo, tolerance, placement, overlap, never-lose-a-segment."""
import json
import os
import subprocess
import unittest

from tests.helpers import TempDirCase, make_cfg, make_tone, needs_ffmpeg

from autodub import timeline
from autodub.dubflow import alignment as al
from autodub.dubflow.alignment import (ALIGNMENT_TOLERANCE, MAX_TTS_SPEED, MIN_TTS_SPEED,
                                       AlignParams, classify, plan_alignment, run_alignment)
from autodub.dubflow.errors import PipelineError
from autodub.dubflow.tts_stage import TtsResult
from autodub.dubflow.validate import audio_duration, read_json
from autodub.dubflow.workspace import Workspace

P = AlignParams()


def recs(*windows):
    return [{"id": i, "start": s, "end": e, "translated_text": "x"}
            for i, (s, e) in enumerate(windows, 1)]


def plan(windows, durations, total=30.0, params=P, classified=True):
    items = plan_alignment(recs(*windows), {i: d for i, d in enumerate(durations, 1)}, total, params)
    if classified:
        for it in items:
            it.audio_path = "x" if it.tts_duration > 0 else None
            it.final_duration = it.tts_duration / it.speed_factor if it.tts_duration else 0.0
        classify(items, total, params)
    return items


class SpecConstantsTest(unittest.TestCase):
    def test_limits_match_the_assignment(self):
        self.assertEqual(MIN_TTS_SPEED, 0.85)
        self.assertEqual(MAX_TTS_SPEED, 1.25)
        self.assertEqual(ALIGNMENT_TOLERANCE, 0.05)
        p = AlignParams.from_cfg(make_cfg())
        self.assertEqual((p.min_speed, p.max_speed, p.tolerance), (0.85, 1.25, 0.05))
        self.assertEqual(p.max_overhang, 0.0)

    def test_invalid_speed_limits_are_rejected(self):
        with self.assertRaises(PipelineError):
            AlignParams.from_cfg(make_cfg(alignment={"max_speed": 3.0}))
        with self.assertRaises(PipelineError):
            AlignParams.from_cfg(make_cfg(alignment={"min_speed": 0.2}))


class TimingCalculationTest(unittest.TestCase):
    def test_shorter_than_window_keeps_natural_speed(self):
        it, = plan([(1.0, 3.0)], [1.5])
        self.assertEqual(it.speed_factor, 1.0)
        self.assertAlmostEqual(it.final_duration, 1.5)
        self.assertEqual(it.status, "ok")

    def test_within_5_percent_tolerance_is_left_alone(self):
        it, = plan([(1.0, 3.0)], [2.08])               # 4% over a 2.0 s window
        self.assertEqual(it.speed_factor, 1.0)
        self.assertEqual(it.status, "ok")

    def test_longer_than_window_gets_speed_factor_tts_over_target(self):
        it, = plan([(1.0, 3.0)], [2.4])
        self.assertAlmostEqual(it.speed_factor, 1.2, places=4)      # 2.4 / 2.0
        self.assertAlmostEqual(it.final_duration, 2.0, places=4)
        self.assertEqual(it.status, "adjusted")

    def test_exactly_at_max_speed_still_fits(self):
        it, = plan([(1.0, 3.0)], [2.5])                # needs exactly 1.25x
        self.assertAlmostEqual(it.speed_factor, 1.25, places=4)
        self.assertEqual(it.status, "adjusted")

    def test_over_max_speed_is_capped_kept_and_flagged_needs_review(self):
        it, = plan([(1.0, 3.0)], [3.0])                # would need 1.5x
        self.assertEqual(it.speed_factor, 1.25)        # capped at the safe maximum
        self.assertAlmostEqual(it.final_duration, 2.4)  # NOT trimmed to 2.0
        self.assertEqual(it.status, "needs_review")
        self.assertIn("longer_than_window_at_max_speed", it.reasons)
        self.assertEqual(it.as_dict()["needs_review"], True)

    def test_min_speed_is_never_breached_when_stretching_short_clips(self):
        off = plan([(1.0, 3.0)], [0.5])[0]
        self.assertEqual(off.speed_factor, 1.0)        # default: natural playback
        on = plan([(1.0, 3.0)], [0.5], params=AlignParams(stretch_short=True))[0]
        self.assertEqual(on.speed_factor, 0.85)        # 0.5/2.0=0.25 clamped to MIN
        near = plan([(1.0, 3.0)], [1.8], params=AlignParams(stretch_short=True))[0]
        self.assertAlmostEqual(near.speed_factor, 0.9, places=4)
        self.assertGreaterEqual(near.speed_factor, 0.85)

    def test_borrowed_silence_lowers_the_needed_speed(self):
        tight = plan([(0.0, 2.0), (6.0, 8.0)], [2.4, 1.0])[0]
        loose = plan([(0.0, 2.0), (6.0, 8.0)], [2.4, 1.0],
                     params=AlignParams(max_overhang=1.0))[0]
        self.assertGreater(tight.speed_factor, 1.0)
        self.assertEqual(loose.speed_factor, 1.0)      # 2.4 fits in 2.0 + 1.0 s of silence


class PlacementAndOverlapTest(unittest.TestCase):
    def test_every_clip_keeps_its_original_start_and_window(self):
        windows = [(0.4, 1.9), (3.25, 5.0), (7.0, 9.5)]
        items = plan(windows, [3.0, 1.0, 2.0])
        for it, (s, e) in zip(items, windows):
            self.assertEqual(it.placed_start, s)       # never moved
            self.assertEqual((it.start, it.end), (s, e))
            self.assertAlmostEqual(it.target_duration, e - s)

    def test_overflow_into_next_segment_is_reported_as_overlap(self):
        a, b = plan([(0.0, 2.0), (2.5, 4.0)], [4.0, 1.0])
        self.assertGreater(a.overlap_next_s, 0.05)      # 4.0/1.25 = 3.2 s > 2.5 s start of b
        self.assertAlmostEqual(a.overlap_next_s, 3.2 - 2.5, places=3)
        self.assertIn("overlaps_next_segment", a.reasons)
        self.assertEqual(a.status, "needs_review")
        self.assertEqual(b.status, "ok")

    def test_source_overlap_alone_is_reported_but_not_a_failure(self):
        a, b = plan([(0.0, 2.0), (1.5, 3.5)], [1.9, 1.0])    # ASR windows overlap already
        self.assertIn("overlaps_next_segment", a.reasons)
        self.assertNotEqual(a.status, "needs_review")

    def test_gaps_between_segments_are_untouched(self):
        items = plan([(0.0, 1.0), (10.0, 11.0)], [0.8, 0.8])
        self.assertEqual(items[1].start - items[0].end, 9.0)
        self.assertEqual([i.overlap_next_s for i in items], [0.0, 0.0])

    def test_clip_running_past_video_end_is_flagged(self):
        it, = plan([(8.0, 9.5)], [4.0], total=10.0)
        self.assertIn("extends_past_video_end", it.reasons)
        self.assertEqual(it.status, "needs_review")

    def test_segment_without_audio_is_not_silently_ok(self):
        items = plan_alignment(recs((0, 1), (2, 3)), {1: 0.5}, 10.0, P)
        items[0].audio_path, items[0].final_duration = "x", 0.5
        items[1].reasons.append("no_audio:tts_failed")
        classify(items, 10.0, P)
        self.assertEqual(items[0].status, "ok")
        self.assertEqual(items[1].status, "needs_review")


class LegacyTimelineUnchangedTest(unittest.TestCase):
    """The tolerance/needs_review additions must not change legacy behaviour.
    Golden values were computed with the ORIGINAL timeline.py (git HEAD) and compared."""
    CASES = [
        (dict(starts=[0, 3, 6], nat=[3.0, 1.0, 4.0], ends=[2, 5, 7],
              kw=dict(max_speed=1.25, min_gap=0.0, total_duration=10, trim_overflow=True, max_overhang=0.0)),
         [[0, 0.0, 1.25, 2.0, True], [1, 3.0, 1.0, 1.0, False], [2, 6.0, 1.25, 1.0, True]]),
        (dict(starts=[0, 3, 6], nat=[3.0, 1.0, 4.0], ends=[2, 5, 7],
              kw=dict(max_speed=1.6, min_gap=0.05, total_duration=10, trim_overflow=True)),
         [[0, 0.0, 1.0909, 2.75, False], [1, 3.0, 1.0, 1.0, False], [2, 6.0, 1.6, 1.75, True]]),
        (dict(starts=[1, 2, 9], nat=[0.0, 5.0, 4.0], ends=[2, 4, 9.5], kw=dict(max_speed=1.5, total_duration=10)),
         [[0, 1.0, 1.0, 0.0, False], [1, 2.0, 1.5, 2.75, True], [2, 9.0, 1.5, 0.96, True]]),
        (dict(starts=[0, 2.5, 5], nat=[1.0, 2.4, 0.5], ends=[2, 4.5, 6], kw=dict()),
         [[0, 0.0, 1.0, 1.0, False], [1, 2.5, 1.0, 2.4, False], [2, 5.0, 1.0, 0.5, False]]),
    ]

    def test_defaults_reproduce_original_results(self):
        for spec, golden in self.CASES:
            got = timeline.fit_segments_strict(spec["starts"], spec["nat"], ends=spec["ends"], **spec["kw"])
            self.assertEqual([[p.index, p.placed_start, p.speed, round(p.final_dur, 6), p.trimmed]
                              for p in got], golden)

    def test_trimmed_legacy_clips_are_now_also_flagged(self):
        spec, _ = self.CASES[0]
        got = timeline.fit_segments_strict(spec["starts"], spec["nat"], ends=spec["ends"], **spec["kw"])
        self.assertTrue(got[0].trimmed and got[0].needs_review)
        self.assertEqual(timeline.summarize(got)["needs_review"], 2)


@needs_ffmpeg
class AtempoTest(TempDirCase):
    def test_atempo_1_25_shortens_audio_by_that_factor(self):
        from autodub.video import change_speed
        src, dst = os.path.join(self.tmp, "a.wav"), os.path.join(self.tmp, "b.wav")
        make_tone(src, 2.5)
        change_speed(src, dst, 1.25)
        self.assertAlmostEqual(audio_duration(dst), 2.0, delta=0.06)

    def test_atempo_preserves_pitch(self):
        from autodub.video import change_speed
        import wave, array
        src, dst = os.path.join(self.tmp, "a.wav"), os.path.join(self.tmp, "b.wav")
        make_tone(src, 2.0, freq=440)
        change_speed(src, dst, 1.25)

        def freq(path):
            with wave.open(path) as w:
                rate = w.getframerate()
                a = array.array("h", w.readframes(w.getnframes()))
            zc = sum(1 for i in range(1, len(a)) if (a[i - 1] < 0) != (a[i] < 0))
            return zc / 2 / (len(a) / rate)
        self.assertAlmostEqual(freq(dst), freq(src), delta=15)      # ~440 Hz both


def make_results(ws, specs):
    out = []
    for sid, dur in specs:
        make_tone(ws.tts_wav(sid), dur)
        out.append(TtsResult(sid, ws.tts_wav(sid), duration=audio_duration(ws.tts_wav(sid))))
    return out


@needs_ffmpeg
class RunAlignmentTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.tmp, "vid").ensure()
        self.cfg = make_cfg()
        self.records = recs((1.0, 3.0), (4.0, 6.0), (8.0, 10.0), (12.0, 14.0))
        # natural / needs 1.2x / far too long / natural
        self.tts = make_results(self.ws, [(1, 1.5), (2, 2.4), (3, 3.5), (4, 1.0)])

    def align(self, atempo=None, force=False, total=16.0):
        return run_alignment(self.ws, self.cfg, self.records, self.tts, total, atempo, force)

    def test_end_to_end_alignment_report(self):
        items, info = self.align()
        by = {i.id: i for i in items}
        self.assertEqual(by[1].status, "ok")
        self.assertEqual(by[2].status, "adjusted")
        self.assertEqual(by[3].status, "needs_review")
        self.assertEqual(by[4].status, "ok")
        self.assertEqual(info["needs_review"], 1)
        self.assertEqual(info["sped_up"], 2)            # 2 (1.2x) and 3 (1.25x cap)
        self.assertEqual(info["max_speed_used"], 1.25)
        self.assertAlmostEqual(audio_duration(by[2].audio_path), 2.0, delta=0.06)
        # the over-long clip is kept whole (3.5 / 1.25 = 2.8 s), never trimmed to 2.0 s
        self.assertAlmostEqual(audio_duration(by[3].audio_path), 2.8, delta=0.08)
        for it in items:
            self.assertEqual(it.placed_start, self.records[it.id - 1]["start"])
            self.assertTrue(os.path.basename(it.audio_path) == f"segment_{it.id:04d}.wav")

    def test_alignment_json_written_with_all_fields(self):
        self.align()
        ok, data = read_json(self.ws.alignment_path, ("segments", "summary", "params"))
        self.assertTrue(ok)
        seg = data["segments"][2]
        for key in ("id", "start", "end", "target_duration", "tts_duration", "speed_factor",
                    "final_duration", "status", "needs_review", "reasons", "overlap_next_s"):
            self.assertIn(key, seg)
        self.assertTrue(seg["needs_review"])
        self.assertEqual(data["params"]["max_speed"], 1.25)

    def test_second_run_is_a_full_cache_hit_and_never_calls_atempo(self):
        self.align()
        calls = []
        items, info = self.align(atempo=lambda s, d, sp: calls.append(sp))
        self.assertEqual(calls, [])
        self.assertEqual(info["cache_hits"], 4)
        self.assertEqual(info["processed"], 0)

    def test_partial_resume_processes_only_missing_or_invalid_clips(self):
        self.align()
        os.remove(self.ws.aligned_wav(2))                       # missing
        with open(self.ws.aligned_wav(3), "wb") as fh:          # corrupted
            fh.write(b"garbage")
        from autodub.video import change_speed
        calls = []

        def spy(s, d, sp):
            calls.append(os.path.basename(d))
            change_speed(s, d, sp)
        items, info = self.align(atempo=spy)
        self.assertEqual(sorted(calls), ["segment_0002.wav", "segment_0003.wav"])
        self.assertEqual(info["cache_hits"], 2)
        self.assertGreater(audio_duration(self.ws.aligned_wav(3)), 2.0)

    def test_changed_tts_input_invalidates_aligned_cache(self):
        self.align()
        make_tone(self.ws.tts_wav(2), 2.9)                       # TTS regenerated, new length
        self.tts[1] = TtsResult(2, self.ws.tts_wav(2), duration=audio_duration(self.ws.tts_wav(2)))
        calls = []
        from autodub.video import change_speed
        self.align(atempo=lambda s, d, sp: (calls.append(sp), change_speed(s, d, sp)))
        self.assertEqual(len(calls), 1)

    def test_changed_window_changes_speed_and_invalidates_only_that_clip(self):
        """Same TTS input, different required speed -> stale aligned clip must be rebuilt."""
        self.align()
        self.records[1] = dict(self.records[1], end=5.6)        # window 2.0 s -> 1.6 s
        from autodub.video import change_speed
        calls = []
        items, info = self.align(atempo=lambda s, d, sp: (calls.append((os.path.basename(d), sp)),
                                                          change_speed(s, d, sp)))
        self.assertEqual([c[0] for c in calls], ["segment_0002.wav"])
        self.assertAlmostEqual(calls[0][1], 1.25, places=4)     # was 1.2, now capped at 1.25
        self.assertEqual(info["cache_hits"], 3)

    def test_atempo_failure_keeps_the_segment_and_flags_it(self):
        def boom(s, d, sp):
            raise RuntimeError("ffmpeg exploded")
        items, info = self.align(atempo=boom)
        by = {i.id: i for i in items}
        self.assertTrue(os.path.isfile(by[2].audio_path))        # not lost
        self.assertEqual(by[2].status, "needs_review")
        self.assertIn("atempo_failed", by[2].reasons)
        self.assertEqual(by[2].speed_factor, 1.0)

    def test_failed_and_skipped_tts_are_flagged_or_skipped_correctly(self):
        self.tts[3] = TtsResult(4, None, failed=True, reason="boom")
        self.tts[0] = TtsResult(1, None, skipped=True, reason="nothing speakable")
        items, info = self.align()
        by = {i.id: i for i in items}
        self.assertEqual(by[4].status, "needs_review")
        self.assertIn("no_audio:tts_failed", by[4].reasons)
        self.assertEqual(by[1].status, "skipped")
        self.assertEqual(len(items), 4)                          # nobody disappears

    def test_overlapping_source_windows_are_handled_without_error(self):
        self.records = recs((0.0, 2.0), (1.0, 3.0), (2.5, 5.0), (6.0, 7.0))
        self.tts = make_results(self.ws, [(1, 1.8), (2, 1.8), (3, 2.0), (4, 0.9)])
        items, info = self.align(total=8.0)
        self.assertEqual(len(items), 4)
        self.assertTrue(all(i.audio_path for i in items))
        self.assertGreaterEqual(info["overlaps"], 1)


if __name__ == "__main__":
    unittest.main()
