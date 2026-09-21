"""12-15 + synthetic end-to-end. Whole pipeline, real ffmpeg, fake ASR/translation/TTS engines.

Verifies: full run, ffprobe validation of the final MP4, video stream COPY, SRT, metrics,
second-run cache hits, partial resume, corrupted-cache recovery, and that a failed stage
is never reported as success.
"""
import json
import os
import shutil
import subprocess
import unittest

from tests.helpers import (FakeASR, FakeSynth, FakeTranslator, TempDirCase, make_cfg,
                           make_video, needs_ffmpeg)

from autodub.dubflow.pipeline import Backends, DubFlowPipeline, STAGE_NAMES
from autodub.dubflow.validate import audio_duration, measure_max_volume_db, probe_media, read_json
from autodub.video import assemble_timeline_audio, change_speed, render_final

SEGS = [(0.5, 3.0, "hola a todos"),
        (3.5, 5.0, "hoy vamos a aprender algo nuevo y muy largo de decir"),   # will need_review
        (6.0, 9.0, "gracias por ver"),
        (9.2, 13.5, "hasta la proxima")]


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *a, **k):
        self.calls.append(a)
        return self.fn(*a, **k)


def video_md5(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy",
                          "-f", "md5", "-"], capture_output=True, text=True).stdout
    return out.strip()


@needs_ffmpeg
class PipelineBase(TempDirCase):
    SECONDS = 14

    def setUp(self):
        super().setUp()
        self.video = make_video(os.path.join(self.tmp, "clip.mp4"), seconds=self.SECONDS)
        self.cfg = make_cfg(data_dir=os.path.join(self.tmp, "data"))
        self.new_backends()

    def new_backends(self, segs=SEGS, lang="es", translator=None):
        self.asr = FakeASR(segs, lang)
        self.tl = translator or FakeTranslator()
        self.synth = FakeSynth()
        self.atempo, self.mixer, self.renderer = Spy(change_speed), Spy(assemble_timeline_audio), Spy(render_final)
        self.backends = Backends(transcribe=self.asr, translator=self.tl, synth=self.synth,
                                 atempo=self.atempo, mixer=self.mixer, renderer=self.renderer)

    def run_pipeline(self, source=None, **kw):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if not os.environ.get("DUBFLOW_TEST_VERBOSE") \
                else contextlib.nullcontext():
            return DubFlowPipeline(self.cfg, backends=self.backends, **kw).run(source or self.video)

    @property
    def inter(self):
        return os.path.join(self.tmp, "data", "intermediate")

    def statuses(self, result):
        return {s["n"]: s["status"] for s in result.stages}


class FullRunTest(PipelineBase):
    def setUp(self):
        super().setUp()
        self.r = self.run_pipeline()

    def test_succeeds_and_reports_all_ten_stages(self):
        self.assertTrue(self.r.ok, self.r.error)
        self.assertEqual(self.r.exit_code, 0)
        self.assertEqual([s["n"] for s in self.r.stages], list(range(1, 11)))
        self.assertEqual(len(STAGE_NAMES), 10)
        self.assertFalse(any(s["status"] == "failed" for s in self.r.stages))

    def test_workspace_layout_matches_the_spec(self):
        vid = self.r.video_id
        base = os.path.join(self.inter, vid)
        for rel in ("tts/segment_0001.wav", "tts/segment_0004.wav", "aligned/segment_0001.wav",
                    "dubbed_audio.wav", "transcript.json", "translations.json", "alignment.json"):
            self.assertTrue(os.path.isfile(os.path.join(base, rel)), rel)
        self.assertEqual(os.path.basename(self.r.final_mp4), f"{vid}_dubbed.mp4")
        self.assertTrue(os.path.isfile(self.r.final_mp4))
        self.assertFalse(os.path.exists(self.r.final_mp4.replace("_dubbed.mp4", "_dubbed.partial.mp4")))

    def test_final_mp4_ffprobe_checks(self):
        info = probe_media(self.r.final_mp4)
        src = probe_media(self.video)
        self.assertGreater(os.path.getsize(self.r.final_mp4), 0)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertEqual(info.video_codec, "h264")
        self.assertEqual(info.audio_codec, "aac")
        self.assertEqual((info.width, info.height), (320, 180))
        self.assertAlmostEqual(info.duration, src.duration, delta=0.15)
        self.assertAlmostEqual(info.video_duration, info.audio_duration, delta=0.3)

    def test_video_stream_is_copied_not_reencoded(self):
        self.assertEqual(video_md5(self.r.final_mp4), video_md5(self.video))

    def test_final_audio_is_the_english_dub_not_the_original_tone(self):
        # The source video carries a constant 220 Hz tone (-18 dB). The final file must hold
        # the dub instead: speech where segments are, true silence in the gap between them.
        def peak_db(t0, dur):
            wav = os.path.join(self.tmp, f"probe_{t0}.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(dur),
                            "-i", self.r.final_mp4, "-vn", wav], check=True)
            return measure_max_volume_db(wav)
        self.assertGreater(peak_db(0.6, 2.0), -40)         # segment 1 is speaking
        self.assertLess(peak_db(7.8, 1.0), -60)            # gap after segment 3: silent
        self.assertLess(peak_db(0.0, 0.4), -60)            # before the first segment: silent
        self.assertAlmostEqual(audio_duration(self.r.dubbed_audio), self.SECONDS, delta=0.3)

    def test_srt_has_english_text_on_the_original_timeline(self):
        with open(self.r.srt_path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("1\n00:00:00,500 --> 00:00:03,000\nEN: hola a todos", text)
        self.assertIn("2\n00:00:03,500 --> 00:00:05,000\n", text)   # NOT stretched to fit the TTS
        self.assertIn("4\n00:00:09,200 --> 00:00:13,500\nEN: hasta la proxima", text)
        self.assertEqual(text.count("-->"), 4)

    def test_metrics_report_contains_the_required_sections(self):
        ok, m = read_json(self.r.metrics_path, ("status",))
        self.assertTrue(ok)
        self.assertEqual(m["status"], "success")
        for key in ("video_id", "source_video", "transcript", "translation", "tts", "alignment",
                    "mix", "final_video", "dubbed_audio", "stages", "needs_review", "voice",
                    "total_seconds", "cache_summary"):
            self.assertIn(key, m)
        self.assertEqual(m["voice"], "en-US-AriaNeural")
        self.assertEqual(m["alignment"]["needs_review"], 1)
        self.assertEqual(m["needs_review"], [2])
        self.assertEqual(m["final_video"]["video_codec"], "h264")
        self.assertTrue(m["final_video"]["checks_passed"])

    def test_alignment_json_marks_the_too_long_segment_needs_review(self):
        _, a = read_json(os.path.join(self.inter, self.r.video_id, "alignment.json"), ("segments",))
        seg2 = a["segments"][1]
        self.assertEqual(seg2["status"], "needs_review")
        self.assertEqual(seg2["speed_factor"], 1.25)
        self.assertEqual(seg2["start"], 3.5)                       # never moved
        self.assertGreater(seg2["final_duration"], seg2["target_duration"])
        self.assertEqual(a["segments"][0]["status"], "ok")

    def test_file_log_records_stages(self):
        log = os.path.join(self.inter, self.r.video_id, "pipeline.log")
        with open(log, encoding="utf-8") as fh:
            text = fh.read()
        for n in (1, 5, 10):
            self.assertIn(f"[{n}/10] {STAGE_NAMES[n]}", text)
        self.assertIn("needs_review", text)
        self.assertIn("cache miss", text.lower())


class CacheAndResumeTest(PipelineBase):
    def test_second_run_reuses_every_expensive_stage(self):
        first = self.run_pipeline()
        self.assertTrue(first.ok)
        asr_calls, tl_calls, synth_calls = self.asr.calls, len(self.tl.calls), len(self.synth.calls)
        self.atempo.calls.clear(); self.mixer.calls.clear(); self.renderer.calls.clear()
        second = self.run_pipeline()
        self.assertTrue(second.ok)
        st = self.statuses(second)
        for n in range(2, 9):
            self.assertEqual(st[n], "cache hit", f"stage {n} {STAGE_NAMES[n]} -> {st[n]}")
        self.assertEqual(self.asr.calls, asr_calls)
        self.assertEqual(len(self.tl.calls), tl_calls)
        self.assertEqual(len(self.synth.calls), synth_calls)
        self.assertEqual((self.atempo.calls, self.mixer.calls, self.renderer.calls), ([], [], []))
        self.assertEqual(first.metrics["voice"], second.metrics["voice"])

    def test_failure_in_translation_keeps_download_and_transcript(self):
        """Stops after transcription -> re-run must not download or transcribe again."""
        dl_calls = []

        def fake_download(url, dest, cfg):
            dl_calls.append(url)
            out = os.path.join(dest, "video.mp4")
            shutil.copyfile(self.video, out)
            return out
        self.backends.download = fake_download

        class Broken(FakeTranslator):
            def translate_batch(self, *a, **k):
                raise RuntimeError("translation service down")
        self.backends.translator = Broken()
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        bad = self.run_pipeline(url)
        self.assertFalse(bad.ok)
        self.assertEqual(bad.exit_code, 1)
        self.assertEqual(bad.failed_stage, "Translating to English")
        self.assertIn("translation service down", bad.error)
        self.assertEqual(bad.video_id, "jNQXAC9IVRw")
        self.assertIsNone(bad.final_mp4)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "data", "output", "jNQXAC9IVRw_dubbed.mp4")))
        _, m = read_json(bad.metrics_path, ("status",))
        self.assertEqual(m["status"], "failed")
        self.assertEqual(m["failed_stage"], "Translating to English")

        self.backends.translator = FakeTranslator()
        good = self.run_pipeline(url)
        self.assertTrue(good.ok, good.error)
        self.assertEqual(dl_calls, [url])                       # downloaded exactly once
        self.assertEqual(self.asr.calls, 1)                     # transcribed exactly once
        st = self.statuses(good)
        self.assertEqual(st[1], "cache hit")
        self.assertEqual(st[3], "cache hit")

    def test_partial_tts_resume_generates_only_missing_clips(self):
        self.run_pipeline()
        vid_dir = os.path.join(self.inter, os.listdir(self.inter)[0])
        before = len(self.synth.calls)
        for sid in (2, 4):
            os.remove(os.path.join(vid_dir, "tts", f"segment_{sid:04d}.wav"))
        r = self.run_pipeline()
        self.assertTrue(r.ok)
        self.assertEqual(len(self.synth.calls) - before, 2)
        self.assertEqual(r.metrics["tts"]["generated"], 2)
        self.assertEqual(r.metrics["tts"]["cache_hits"], 2)

    def test_partial_aligned_resume_processes_only_missing_clips(self):
        self.run_pipeline()
        vid_dir = os.path.join(self.inter, os.listdir(self.inter)[0])
        for sid in (1, 3):
            os.remove(os.path.join(vid_dir, "aligned", f"segment_{sid:04d}.wav"))
        r = self.run_pipeline()
        self.assertTrue(r.ok)
        self.assertEqual(r.metrics["alignment"]["processed"], 2)
        self.assertEqual(r.metrics["alignment"]["cache_hits"], 2)

    def test_corrupted_intermediate_files_are_regenerated(self):
        self.run_pipeline()
        vid_dir = os.path.join(self.inter, os.listdir(self.inter)[0])
        with open(os.path.join(vid_dir, "transcript.json"), "w") as fh:
            fh.write("{ broken")
        with open(os.path.join(vid_dir, "audio_16k.wav"), "wb") as fh:
            fh.write(b"not audio")
        self.renderer.calls.clear()
        r = self.run_pipeline()
        self.assertTrue(r.ok, r.error)
        st = self.statuses(r)
        self.assertEqual(st[2], "done")           # audio re-extracted
        self.assertEqual(st[3], "done")           # transcript redone
        self.assertEqual(self.asr.calls, 2)
        self.assertGreater(audio_duration(os.path.join(vid_dir, "audio_16k.wav")), 10)

    def test_corrupted_final_mp4_is_rerendered(self):
        r1 = self.run_pipeline()
        with open(r1.final_mp4, "wb") as fh:
            fh.write(b"\x00" * 1000)
        self.renderer.calls.clear()
        r2 = self.run_pipeline()
        self.assertTrue(r2.ok, r2.error)
        self.assertEqual(len(self.renderer.calls), 1)
        self.assertTrue(probe_media(r2.final_mp4).has_audio)

    def test_final_video_without_audio_stream_is_not_accepted_from_cache(self):
        r1 = self.run_pipeline()
        silent = os.path.join(self.tmp, "silent.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", r1.final_mp4, "-an", "-c:v", "copy", silent], check=True)
        shutil.copyfile(silent, r1.final_mp4)
        self.renderer.calls.clear()
        r2 = self.run_pipeline()
        self.assertTrue(r2.ok)
        self.assertEqual(len(self.renderer.calls), 1)
        self.assertTrue(probe_media(r2.final_mp4).has_audio)

    def test_redo_forces_only_the_named_stage_and_force_forces_all(self):
        self.run_pipeline()
        before = len(self.synth.calls)
        r = self.run_pipeline(redo=("tts",))
        self.assertTrue(r.ok)
        self.assertEqual(len(self.synth.calls) - before, 4)     # all clips rebuilt
        self.assertEqual(self.asr.calls, 1)                     # transcript untouched
        r = self.run_pipeline(force=True)
        self.assertEqual(self.asr.calls, 2)
        self.assertEqual(self.statuses(r)[3], "done")

    def test_unknown_redo_stage_is_rejected(self):
        from autodub.dubflow.errors import PipelineError
        with self.assertRaises(PipelineError):
            DubFlowPipeline(self.cfg, redo=("nope",))


class FailureHonestyTest(PipelineBase):
    def test_render_crash_is_reported_as_failure_and_leaves_no_final_file(self):
        def crash(*a, **k):
            raise RuntimeError("ffmpeg died")
        self.backends.renderer = crash
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertEqual(r.exit_code, 1)
        self.assertEqual(r.failed_stage, "Rendering final video")
        self.assertIsNone(r.final_mp4)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "data", "output", f"{r.video_id}_dubbed.mp4")))
        _, m = read_json(r.metrics_path, ("status",))
        self.assertEqual(m["status"], "failed")
        self.assertNotEqual([s["status"] for s in r.stages][-1], "done")

    def test_render_that_produces_a_video_without_audio_fails_validation(self):
        def bad_renderer(video, dub, out, **k):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-an", "-c:v", "copy", out], check=True)
            return out
        self.backends.renderer = bad_renderer
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_stage, "Rendering final video")
        self.assertIn("no audio stream", r.error)
        out_dir = os.path.join(self.tmp, "data", "output")
        self.assertEqual([f for f in os.listdir(out_dir) if f.endswith(".mp4")], [])   # partial cleaned

    def test_render_with_wrong_duration_fails_validation(self):
        def short_renderer(video, dub, out, **k):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-t", "5", "-c", "copy", out], check=True)
            return out
        self.backends.renderer = short_renderer
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertIn("duration", r.error)

    def test_missing_input_file_fails_cleanly(self):
        r = self.run_pipeline(os.path.join(self.tmp, "nope.mp4"))
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_stage, "Downloading video")
        self.assertIn("not found", r.error.lower())

    def test_video_without_audio_is_rejected_up_front(self):
        silent = make_video(os.path.join(self.tmp, "mute.mp4"), seconds=4, audio=False)
        r = self.run_pipeline(silent)
        self.assertFalse(r.ok)
        self.assertIn("audio", r.error)

    def test_no_speech_is_an_error_not_an_empty_success(self):
        self.backends.transcribe = FakeASR([], "en")
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_stage, "Transcribing")
        self.assertIn("No speech", r.error)

    def test_total_tts_failure_stops_before_rendering(self):
        self.backends.synth = FakeSynth(fail_texts=[f"EN: {t}" for _, _, t in SEGS])
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_stage, "Generating English TTS")
        self.assertEqual(self.renderer.calls, [])

    def test_partial_tts_failure_still_produces_video_but_flags_the_segment(self):
        self.backends.synth = FakeSynth(fail_texts=["EN: gracias por ver"])
        r = self.run_pipeline()
        self.assertTrue(r.ok)
        self.assertIn(3, r.metrics["needs_review"])
        self.assertEqual(r.metrics["tts"]["failed"], [3])
        _, a = read_json(os.path.join(self.inter, r.video_id, "alignment.json"), ("segments",))
        self.assertIn("no_audio:tts_failed", a["segments"][2]["reasons"])


class EnglishSourceTest(PipelineBase):
    def test_english_video_uses_passthrough_without_any_translation_model(self):
        self.new_backends(segs=[(0.5, 3.0, "Hello everyone"), (3.5, 6.0, "this is a test")], lang="en")
        self.backends.translator = None                # let the real provider selection run
        r = self.run_pipeline()
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.metrics["translation"]["provider"], "passthrough")
        with open(r.srt_path, encoding="utf-8") as fh:
            self.assertIn("Hello everyone", fh.read())

    def test_non_english_default_provider_fails_clearly_if_local_deps_missing(self):
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch installed; real model would be loaded")
        self.backends.translator = None
        r = self.run_pipeline()
        self.assertFalse(r.ok)
        self.assertEqual(r.failed_stage, "Translating to English")
        self.assertIn("pip install transformers sentencepiece torch", r.error)


class SourceVariantsTest(PipelineBase):
    """Render branches and path handling that the default H.264 test video never reaches."""

    def _transcode(self, src, name, *codec_args):
        out = os.path.join(self.tmp, name)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *codec_args, out],
                       check=True, stdin=subprocess.DEVNULL)
        return out

    def test_non_h264_source_is_reencoded_to_h264_by_default(self):
        src = self._transcode(self.video, "mpeg4_source.mp4", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac")
        self.assertEqual(probe_media(src).video_codec, "mpeg4")
        r = self.run_pipeline(src)
        self.assertTrue(r.ok, r.error)
        info = probe_media(r.final_mp4)
        self.assertEqual((info.video_codec, info.audio_codec), ("h264", "aac"))
        self.assertNotEqual(video_md5(r.final_mp4), video_md5(src))     # really re-encoded
        self.assertAlmostEqual(info.duration, probe_media(src).duration, delta=0.2)

    def test_force_h264_false_keeps_the_original_codec_by_copying(self):
        src = self._transcode(self.video, "mpeg4_source.mp4", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac")
        self.cfg["render"]["force_h264"] = False
        r = self.run_pipeline(src)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(probe_media(r.final_mp4).video_codec, "mpeg4")
        self.assertEqual(video_md5(r.final_mp4), video_md5(src))        # stream copy

    def test_path_with_spaces_and_non_ascii_characters(self):
        folder = os.path.join(self.tmp, "Mis vídeos (nuevos)")
        os.makedirs(folder)
        src = os.path.join(folder, "clase de matemáticas #1.mp4")
        shutil.copyfile(self.video, src)
        r = self.run_pipeline(src)
        self.assertTrue(r.ok, r.error)
        self.assertTrue(os.path.isfile(r.final_mp4))
        self.assertRegex(r.video_id, r"^[A-Za-z0-9._-]+$")              # safe folder name
        with open(r.srt_path, encoding="utf-8") as fh:
            self.assertIn("EN: hola a todos", fh.read())

    def test_video_whose_audio_is_shorter_than_its_video_still_dubs(self):
        # audio 10 s, video 14 s: a common real-world mismatch
        odd = os.path.join(self.tmp, "short_audio.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=15:d=14",
                        "-f", "lavfi", "-i", "sine=f=220:d=10", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", odd], check=True, stdin=subprocess.DEVNULL)
        r = self.run_pipeline(odd)
        self.assertTrue(r.ok, r.error)
        info = probe_media(r.final_mp4)
        self.assertAlmostEqual(info.duration, probe_media(odd).duration, delta=0.3)
        self.assertAlmostEqual(info.video_duration, info.audio_duration, delta=0.3)


class ManySegmentsTest(PipelineBase):
    """Scale smoke test: 40 segments over a 60 s video, then a full-cache second run."""
    SECONDS = 60

    def test_forty_segments_resume_without_regenerating(self):
        segs = [(i * 1.5 + 0.2, i * 1.5 + 1.3, f"line number {i}") for i in range(40)]
        self.new_backends(segs=segs, lang="en")
        self.backends.translator = None
        r1 = self.run_pipeline()
        self.assertTrue(r1.ok, r1.error)
        self.assertEqual(r1.metrics["tts"]["generated"], 40)
        calls = len(self.synth.calls)
        r2 = self.run_pipeline()
        self.assertTrue(r2.ok)
        self.assertEqual(len(self.synth.calls), calls)
        self.assertEqual(r2.metrics["alignment"]["cache_hits"], 40)
        self.assertEqual(self.statuses(r2)[8], "cache hit")


if __name__ == "__main__":
    unittest.main()
