"""Workspace naming, video ids, download helpers, final-MP4 validator, SRT, CLI."""
import inspect
import os
import subprocess
import unittest
from unittest import mock

from tests.helpers import ROOT, TempDirCase, ff, make_cfg, make_video, needs_ffmpeg

import main as cli
from autodub import downloader
from autodub.dubflow import pipeline as pl
from autodub.dubflow.errors import ValidationError
from autodub.dubflow.report import write_metrics, write_srt
from autodub.dubflow.validate import (audio_duration, probe_media, validate_final_mp4,
                                      wav_duration)
from autodub.dubflow.workspace import Workspace, derive_video_id, is_url
from autodub.srt_utils import load_srt_file


class VideoIdAndLayoutTest(unittest.TestCase):
    def test_youtube_url_forms_map_to_the_same_id(self):
        for url in ("https://www.youtube.com/watch?v=jNQXAC9IVRw",
                    "https://youtube.com/watch?v=jNQXAC9IVRw&list=PL1&index=3&t=9s",
                    "https://youtu.be/jNQXAC9IVRw?si=abc",
                    "https://www.youtube.com/shorts/jNQXAC9IVRw",
                    "https://www.youtube.com/embed/jNQXAC9IVRw",
                    "https://m.youtube.com/watch?v=jNQXAC9IVRw"):
            self.assertEqual(derive_video_id(url), "jNQXAC9IVRw", url)

    def test_other_urls_and_files_get_stable_safe_ids(self):
        a = derive_video_id("https://vimeo.com/12345")
        self.assertEqual(a, derive_video_id("https://vimeo.com/12345"))
        self.assertNotEqual(a, derive_video_id("https://vimeo.com/99999"))
        f = derive_video_id("/tmp/My Video (final) ü.mp4")
        self.assertRegex(f, r"^[A-Za-z0-9._-]+$")               # safe as a folder name
        self.assertEqual(f, derive_video_id("/tmp/My Video (final) ü.mp4"))
        self.assertNotEqual(f, derive_video_id("/other/My Video (final) ü.mp4"))
        self.assertTrue(is_url("https://x.y") and not is_url("C:\\a.mp4"))

    def test_paths_match_the_documented_layout(self):
        ws = Workspace("data", "abc123")
        p = lambda x: x.replace(os.sep, "/")
        self.assertEqual(p(ws.tts_wav(1)), "data/intermediate/abc123/tts/segment_0001.wav")
        self.assertEqual(p(ws.tts_wav(123)), "data/intermediate/abc123/tts/segment_0123.wav")
        self.assertEqual(p(ws.dubbed_audio_path), "data/intermediate/abc123/dubbed_audio.wav")
        self.assertEqual(p(ws.final_mp4), "data/output/abc123_dubbed.mp4")
        self.assertEqual(p(ws.srt_path), "data/output/abc123_en.srt")
        self.assertNotIn("vietsub", ws.final_mp4)


class DownloadHelpersTest(TempDirCase):
    def test_format_selector_prefers_h264_and_respects_height(self):
        s = pl.build_format_selector({"quality": "720", "prefer_h264": True})
        self.assertIn("[height<=720]", s)
        self.assertIn("vcodec^=avc1", s)
        self.assertIn("ba[ext=m4a]", s)
        self.assertTrue(s.rstrip().endswith("[vcodec!=none]"))       # always has a last-resort fallback
        self.assertNotIn("height", pl.build_format_selector({"quality": "best", "prefer_h264": False}))
        self.assertNotIn("avc1", pl.build_format_selector({"quality": "480", "prefer_h264": False}))

    def test_download_video_passes_format_selector_to_ytdlp(self):
        self.assertIn("format_selector", inspect.signature(downloader.download_video).parameters)

        class Stop(Exception):
            pass
        seen = {}

        def capture(url, out_tmpl, fmt, *a, **k):
            seen["fmt"] = fmt
            raise Stop

        with mock.patch.object(downloader, "_build_download_command", capture):
            with self.assertRaises(Stop):
                downloader.download_video("https://www.youtube.com/watch?v=jNQXAC9IVRw", self.tmp,
                                          quality="720", format_selector="MY-SELECTOR")
            self.assertEqual(seen["fmt"], "MY-SELECTOR")
            with self.assertRaises(Stop):                             # legacy behaviour untouched
                downloader.download_video("https://www.youtube.com/watch?v=jNQXAC9IVRw", self.tmp,
                                          quality="720")
            self.assertEqual(seen["fmt"], downloader._QUALITY["720"])

    @needs_ffmpeg
    def test_find_cached_source_only_accepts_complete_valid_videos(self):
        d = os.path.join(self.tmp, "source")
        os.makedirs(d)
        self.assertIsNone(pl.find_cached_source(d))
        self.assertIsNone(pl.find_cached_source(os.path.join(self.tmp, "missing")))
        open(os.path.join(d, "half.mp4.part"), "wb").write(b"x" * 500)          # yt-dlp partial
        open(os.path.join(d, "junk.mp4"), "wb").write(b"not a video")           # corrupt
        make_video(os.path.join(d, "mute.mp4"), seconds=2, audio=False)         # no audio stream
        self.assertIsNone(pl.find_cached_source(d))
        good = make_video(os.path.join(d, "good.mp4"), seconds=2)
        self.assertEqual(pl.find_cached_source(d), good)


@needs_ffmpeg
class FinalMp4ValidationTest(TempDirCase):
    def test_valid_file_passes_and_reports_codecs(self):
        p = make_video(os.path.join(self.tmp, "ok.mp4"), seconds=6)
        rep = validate_final_mp4(p, 6.0)
        self.assertTrue(rep["checks_passed"])
        self.assertEqual((rep["video_codec"], rep["audio_codec"]), ("h264", "aac"))
        self.assertEqual((rep["width"], rep["height"]), (320, 180))

    def test_each_kind_of_broken_file_is_rejected_with_a_reason(self):
        ok = make_video(os.path.join(self.tmp, "ok.mp4"), seconds=6)
        cases = {}
        cases["no audio stream"] = make_video(os.path.join(self.tmp, "a.mp4"), seconds=6, audio=False)
        audio_only = os.path.join(self.tmp, "b.m4a")
        ff("-f", "lavfi", "-i", "sine=d=6", "-c:a", "aac", audio_only)
        cases["no video stream"] = audio_only
        for reason, path in cases.items():
            with self.assertRaises(ValidationError, msg=reason) as cm:
                validate_final_mp4(path, 6.0)
            self.assertIn(reason, str(cm.exception))
        with self.assertRaises(ValidationError) as cm:                # wrong overall length
            validate_final_mp4(ok, 20.0)
        self.assertIn("differs from source", str(cm.exception))
        mismatch = os.path.join(self.tmp, "av.mp4")                   # audio 3 s, video 10 s
        ff("-f", "lavfi", "-i", "testsrc2=s=160x90:r=10:d=10", "-f", "lavfi", "-i", "sine=d=3",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", mismatch)
        with self.assertRaises(ValidationError) as cm:
            validate_final_mp4(mismatch, 10.0)
        self.assertIn("audio", str(cm.exception))
        empty = os.path.join(self.tmp, "empty.mp4")
        open(empty, "wb").close()
        with self.assertRaises(ValidationError) as cm:
            validate_final_mp4(empty, 6.0)
        self.assertIn("empty", str(cm.exception))
        junk = os.path.join(self.tmp, "junk.mp4")
        open(junk, "wb").write(b"definitely not an mp4" * 50)
        with self.assertRaises(ValidationError):
            validate_final_mp4(junk, 6.0)
        with self.assertRaises(ValidationError) as cm:
            validate_final_mp4(os.path.join(self.tmp, "nope.mp4"), 6.0)
        self.assertIn("does not exist", str(cm.exception))

    def test_wav_duration_uses_stdlib_and_rejects_garbage(self):
        p = os.path.join(self.tmp, "t.wav")
        ff("-f", "lavfi", "-i", "sine=d=1.5:sample_rate=24000", "-c:a", "pcm_s16le", p)
        self.assertAlmostEqual(wav_duration(p), 1.5, places=2)
        open(os.path.join(self.tmp, "g.wav"), "wb").write(b"RIFF nope")
        self.assertIsNone(wav_duration(os.path.join(self.tmp, "g.wav")))
        self.assertEqual(audio_duration(os.path.join(self.tmp, "g.wav")), 0.0)
        self.assertEqual(audio_duration(os.path.join(self.tmp, "missing.wav")), 0.0)


class SrtTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.tmp, "vid").ensure()

    def rec(self, i, s, e, text):
        return {"id": i, "start": s, "end": e, "source_text": "src", "translated_text": text}

    def test_english_srt_exact_format_and_original_millisecond_timestamps(self):
        path = write_srt(self.ws, [self.rec(1, 1.0, 3.5, "Hello everyone."),
                                   self.rec(2, 3.5, 6.2, "Today we are going to learn something new."),
                                   self.rec(3, 3661.007, 3662.5, "One hour in.")])
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(text.strip(), (
            "1\n00:00:01,000 --> 00:00:03,500\nHello everyone.\n\n"
            "2\n00:00:03,500 --> 00:00:06,200\nToday we are going to learn something new.\n\n"
            "3\n01:01:01,007 --> 01:01:02,500\nOne hour in."))

    def test_srt_round_trips_and_skips_empty_translations_with_sequential_numbers(self):
        path = write_srt(self.ws, [self.rec(1, 0.0, 1.0, "First."), self.rec(2, 1.0, 2.0, ""),
                                   self.rec(3, 2.0, 3.0, "Third café.")])
        segs = load_srt_file(path)
        self.assertEqual([(s.index, s.text) for s in segs], [(1, "First."), (2, "Third café.")])
        self.assertEqual((segs[1].start, segs[1].end), (2.0, 3.0))

    def test_metrics_file_is_valid_json_with_timestamp(self):
        import json
        path = write_metrics(self.ws, {"status": "success", "x": 1})
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["x"], 1)
        self.assertIn("generated_at", data)


class CliTest(TempDirCase):
    def test_parser_and_overrides(self):
        a = cli.build_parser().parse_args(
            ["https://youtu.be/jNQXAC9IVRw", "--voice", "en-US-GuyNeural", "--model", "base",
             "--provider", "passthrough", "--language", "es", "--quality", "480",
             "--data-dir", "D", "--redo", "tts,align", "--force"])
        cfg = cli.apply_overrides(make_cfg(), a)
        self.assertEqual(cfg["tts"]["voice"], "en-US-GuyNeural")
        self.assertEqual(cfg["asr"]["model_size"], "base")
        self.assertEqual(cfg["asr"]["language"], "es")
        self.assertEqual(cfg["translation"]["provider"], "passthrough")
        self.assertEqual((cfg["download"]["quality"], cfg["data_dir"]), ("480", "D"))
        self.assertTrue(a.force)

    def test_missing_argument_exits_with_usage_error(self):
        with self.assertRaises(SystemExit) as cm, mock.patch("sys.stderr"):
            cli.build_parser().parse_args([])
        self.assertEqual(cm.exception.code, 2)

    def test_bad_config_and_bad_redo_return_exit_code_1(self):
        with mock.patch("sys.stderr"):
            self.assertEqual(cli.main(["x.mp4", "--config", os.path.join(self.tmp, "nope.yaml")]), 1)
            self.assertEqual(cli.main(["x.mp4", "--redo", "bogus"]), 1)

    def test_missing_input_returns_exit_code_1(self):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli.main([os.path.join(self.tmp, "missing.mp4"), "--data-dir", self.tmp])
        self.assertEqual(code, 1)

    @needs_ffmpeg
    def test_help_runs_as_a_script(self):
        out = subprocess.run(["python3", os.path.join(ROOT, "main.py"), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("YouTube URL", out.stdout)
        self.assertIn("--redo", out.stdout)


class ToolCheckTest(unittest.TestCase):
    def test_missing_ffmpeg_is_a_clear_error(self):
        from autodub.dubflow.errors import PipelineError
        with mock.patch.object(pl, "which", return_value=None):
            with self.assertRaises(PipelineError) as cm:
                pl.check_tools()
        self.assertIn("FFmpeg", str(cm.exception))
        self.assertIn("PATH", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
