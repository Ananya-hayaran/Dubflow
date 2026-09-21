"""9-11. Timeline placement, preserved silence, overlap handling, clipping, duration, mix cache."""
import array
import os
import subprocess
import unittest
import wave

from tests.helpers import TempDirCase, make_cfg, make_tone, needs_ffmpeg, ff

from autodub.dubflow import assemble
from autodub.dubflow.alignment import AlignmentItem
from autodub.dubflow.errors import PipelineError
from autodub.dubflow.validate import audio_duration, read_json
from autodub.dubflow.workspace import Workspace


def load_pcm(path):
    """(rate, channel-0 samples as array('h')) of a 16-bit WAV."""
    with wave.open(path) as w:
        ch, rate = w.getnchannels(), w.getframerate()
        data = array.array("h", w.readframes(w.getnframes()))
    return rate, data[::ch]


def active_ranges(path, threshold=800, win=0.01):
    """Merged (start, end) seconds where the signal is above ``threshold``."""
    rate, s = load_pcm(path)
    step = int(rate * win)
    ranges, cur = [], None
    for i in range(0, len(s), step):
        loud = max(abs(v) for v in s[i:i + step]) > threshold if s[i:i + step] else False
        t = i / rate
        if loud and cur is None:
            cur = t
        elif not loud and cur is not None:
            ranges.append((cur, t))
            cur = None
    if cur is not None:
        ranges.append((cur, len(s) / rate))
    return ranges


def peak(path, start=0.0, end=None):
    rate, s = load_pcm(path)
    a, b = int(start * rate), int(end * rate) if end else len(s)
    return max((abs(v) for v in s[a:b]), default=0)


def make_loud(path, seconds, freq=440):
    ff("-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}:sample_rate=24000,volume=7.2",
       "-ac", "1", "-c:a", "pcm_s16le", path)


@needs_ffmpeg
class MixTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.tmp, "vid").ensure()
        self.cfg = make_cfg()

    def item(self, sid, start, dur, loud=False):
        path = self.ws.aligned_wav(sid)
        (make_loud if loud else make_tone)(path, dur)
        return AlignmentItem(id=sid, start=start, end=start + dur, target_duration=dur,
                             tts_duration=dur, final_duration=audio_duration(path), audio_path=path)

    def mix(self, items, total=10.0, **kw):
        return assemble.run_mix(self.ws, self.cfg, items, total, **kw)

    # ---- placement / silence / duration ----
    def test_clips_are_placed_at_their_original_timestamps(self):
        items = [self.item(1, 1.0, 1.0), self.item(2, 4.5, 0.5), self.item(3, 8.0, 1.5)]
        path, key, info = self.mix(items)
        ranges = active_ranges(path)
        self.assertEqual(len(ranges), 3)
        for (a, b), it in zip(ranges, items):
            self.assertAlmostEqual(a, it.start, delta=0.03)            # onset at original start
            self.assertAlmostEqual(b - a, it.final_duration, delta=0.05)

    def test_it_is_not_a_concatenation_silence_between_segments_is_preserved(self):
        items = [self.item(1, 1.0, 1.0), self.item(2, 6.0, 1.0)]
        path, _, _ = self.mix(items)
        self.assertEqual(peak(path, 2.1, 5.9), 0)                      # dead silence in the gap
        self.assertEqual(peak(path, 0.0, 0.95), 0)                     # and before the first
        self.assertEqual(peak(path, 7.1, 9.9), 0)                      # and after the last
        total_active = sum(b - a for a, b in active_ranges(path))
        self.assertLess(total_active, 2.2)                             # 2 s of speech in 10 s

    def test_dubbed_audio_matches_video_duration(self):
        path, _, info = self.mix([self.item(1, 1.0, 1.0)], total=10.0)
        self.assertAlmostEqual(audio_duration(path), 10.0, delta=0.3)
        self.assertEqual(os.path.basename(path), "dubbed_audio.wav")
        self.assertEqual(os.path.dirname(path), self.ws.root)

    def test_clip_crossing_a_chunk_boundary_stays_in_place(self):
        """Long-video path: mixed in 3 s windows; a clip spanning 2.5-4.0 s must not break."""
        self.cfg["mix"]["chunk_seconds"] = 3
        items = [self.item(1, 0.5, 1.0), self.item(2, 2.5, 1.5), self.item(3, 7.0, 1.0)]
        path, _, _ = self.mix(items)
        ranges = active_ranges(path)
        self.assertEqual(len(ranges), 3)
        self.assertAlmostEqual(ranges[1][0], 2.5, delta=0.03)
        self.assertAlmostEqual(ranges[1][1] - ranges[1][0], 1.5, delta=0.06)   # not cut at 3.0 s
        self.assertAlmostEqual(ranges[2][0], 7.0, delta=0.03)

    # ---- overlap / clipping ----
    def test_overlapping_clips_are_both_audible_and_do_not_clip(self):
        items = [self.item(1, 2.0, 2.0, loud=True), self.item(2, 3.0, 2.0, loud=True)]
        path, _, _ = self.mix(items)
        _, s = load_pcm(path)
        clipped = sum(1 for v in s if abs(v) >= 32767)
        self.assertEqual(clipped, 0, "overlap of two loud clips hard-clipped")
        self.assertLess(peak(path), 32768 * 0.97)
        self.assertGreater(peak(path, 3.2, 3.8), 8000)                # overlap region audible
        self.assertGreater(peak(path, 2.1, 2.9), 8000)                # first-only region audible
        self.assertGreater(peak(path, 4.2, 4.9), 8000)                # second-only region audible

    def test_limiter_is_transparent_for_normal_level_speech(self):
        """A one-clip mix has no limiter in its ffmpeg graph; a multi-clip mix does.
        Their level for the same clip must be identical (limiter adds no attenuation).
        (Both are 3 dB below the raw mono clip: that is the pre-existing mono->stereo
        conversion in the original mixer, verified identical on the original code.)"""
        first = self.item(1, 1.0, 1.0)
        raw_peak = peak(first.audio_path)
        solo_path, _, _ = self.mix([first])                    # no amix/limiter
        solo = peak(solo_path, 0.9, 2.1)
        multi_path, _, _ = self.mix([first, self.item(2, 5.0, 1.0)])   # amix + limiter
        self.assertEqual(peak(multi_path, 0.9, 2.1), solo)
        self.assertAlmostEqual(solo / raw_peak, 0.7071, delta=0.01)

    # ---- safety ----
    def test_no_clips_refuses_to_produce_a_silent_video(self):
        with self.assertRaises(PipelineError) as cm:
            self.mix([AlignmentItem(id=1, start=0, end=1, target_duration=1)])
        self.assertIn("no dubbed audio clips", str(cm.exception))

    def test_silent_mix_is_detected(self):
        def silent_mixer(clips, starts, total, out, **kw):
            ff("-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo", "-t", str(total + 0.2),
               "-c:a", "pcm_s16le", out)
            return out
        with self.assertRaises(PipelineError) as cm:
            self.mix([self.item(1, 1.0, 1.0)], mixer=silent_mixer)
        self.assertIn("silent", str(cm.exception))

    def test_wrong_length_mix_is_rejected(self):
        def short_mixer(clips, starts, total, out, **kw):
            make_tone(out, 3.0)
            return out
        with self.assertRaises(PipelineError) as cm:
            self.mix([self.item(1, 1.0, 1.0)], mixer=short_mixer)
        self.assertIn("does not match", str(cm.exception))

    # ---- cache ----
    def test_second_mix_is_a_cache_hit(self):
        items = [self.item(1, 1.0, 1.0), self.item(2, 5.0, 1.0)]
        self.mix(items)
        calls = []
        _, _, info = self.mix(items, mixer=lambda *a, **k: calls.append(1))
        self.assertTrue(info["cache_hit"])
        self.assertEqual(calls, [])

    def test_changed_clip_or_position_or_corrupt_output_triggers_remix(self):
        items = [self.item(1, 1.0, 1.0), self.item(2, 5.0, 1.0)]
        path, key1, _ = self.mix(items)
        # 1) a clip's audio changed on disk
        make_tone(items[1].audio_path, 1.4)
        items[1].final_duration = audio_duration(items[1].audio_path)
        _, key2, info = self.mix(items)
        self.assertFalse(info["cache_hit"])
        self.assertNotEqual(key1, key2)
        # 2) a clip moved
        items[0].start = 1.5
        _, key3, info = self.mix(items)
        self.assertFalse(info["cache_hit"])
        # 3) the dubbed file got corrupted
        with open(path, "wb") as fh:
            fh.write(b"RIFF garbage")
        _, _, info = self.mix(items)
        self.assertFalse(info["cache_hit"])
        self.assertGreater(audio_duration(path), 9.0)
        # 4) metadata deleted
        os.remove(self.ws.dubbed_meta_path)
        _, _, info = self.mix(items)
        self.assertFalse(info["cache_hit"])


if __name__ == "__main__":
    unittest.main()
