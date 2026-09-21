"""The fake engines used elsewhere bypass the REAL call sites. These tests run DubFlow's real
wrapper code, but with the reused function replaced by a recorder that binds the arguments
against the function's true signature - so a wrong/renamed keyword fails here, not in a
real run on someone's machine. No network, no models."""
import asyncio
import inspect
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

from tests.helpers import TempDirCase, make_cfg

from autodub import asr, downloader, translate, tts
from autodub.dubflow import pipeline as pl
from autodub.dubflow import transcript as tr_stage
from autodub.dubflow import translation as tl
from autodub.dubflow import tts_stage
from autodub.srt_utils import Segment


def binding_recorder(real, result, calls):
    """Callable that validates its call against ``real``'s signature, records it, returns result."""
    sig = inspect.signature(real)

    def rec(*a, **k):
        bound = sig.bind(*a, **k)          # TypeError if any keyword/arity is wrong
        bound.apply_defaults()
        calls.append(bound.arguments)
        return result
    return rec


class TranscribeCallSiteTest(unittest.TestCase):
    def test_default_transcriber_calls_asr_transcribe_with_valid_arguments(self):
        calls = []
        segs = [Segment(index=0, start=0.5, end=2.0, text="Hello there"), ]
        fake = binding_recorder(asr.transcribe, (segs, "en"), calls)
        cfg = make_cfg(asr={"device": "cpu", "compute_type": "int8", "model_size": "base",
                            "language": "es"})
        with mock.patch.object(asr, "transcribe", fake):
            out, lang = tr_stage._default_transcriber("/x/audio.wav", cfg)
        self.assertEqual((out, lang), (segs, "en"))
        a = calls[0]
        self.assertEqual(a["backend"], "faster-whisper")
        self.assertEqual((a["device"], a["compute_type"], a["model_size"]), ("cpu", "int8", "base"))
        self.assertEqual(a["language"], "es")
        self.assertIsNone(a["fallback_backend"])           # no Chinese-only fallback

    def test_real_asr_segment_objects_flow_through_build_segments(self):
        raw = [Segment(index=7, start=0.0, end=1.5, text="  Hello   there "),
               Segment(index=8, start=2.0, end=2.01, text="tiny"),        # too short: widened
               Segment(index=9, start=3.0, end=4.0, text="♪ ♪")]          # not speakable: dropped
        rows = tr_stage.build_segments(raw, duration=10.0)
        self.assertEqual([r["id"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["text"], "Hello there")
        self.assertAlmostEqual(rows[1]["end"] - rows[1]["start"], 0.05, places=6)   # widened

    def test_asr_language_codes_drive_the_passthrough_decision(self):
        self.assertTrue(tl.is_english("en"))
        self.assertTrue(tl.is_english("EN-us"))
        self.assertFalse(tl.is_english("es"))
        self.assertFalse(tl.is_english("auto"))            # unknown -> not passthrough
        with self.assertRaises(tl.TranslationError):
            tl.to_nllb_code("auto")


class DownloadCallSiteTest(unittest.TestCase):
    def test_default_download_calls_download_video_with_valid_arguments(self):
        calls = []
        fake = binding_recorder(downloader.download_video, "/x/video.mp4", calls)
        cfg = make_cfg(download={"quality": "480", "cookies_from_browser": "chrome"})
        with mock.patch.object(downloader, "download_video", fake):
            path = pl._default_download("https://youtu.be/jNQXAC9IVRw", "/x", cfg)
        self.assertEqual(path, "/x/video.mp4")
        a = calls[0]
        self.assertEqual(a["url"], "https://youtu.be/jNQXAC9IVRw")
        self.assertEqual(a["out_dir"], "/x")
        self.assertEqual(a["cookies_from_browser"], "chrome")
        self.assertIn("[height<=480]", a["format_selector"])
        self.assertIn("avc1", a["format_selector"])


class EdgeTtsCallSiteTest(TempDirCase):
    def test_edge_synth_calls_synth_one_with_valid_arguments_from_a_worker_thread(self):
        calls = []
        sig = inspect.signature(tts._synth_one)

        async def fake_synth_one(*a, **k):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            calls.append(bound.arguments)
            return True

        cfg = make_cfg(tts={"max_retries": 2, "retry_delay": 0.5})
        with mock.patch.object(tts_stage.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(tts, "_synth_one", fake_synth_one):
            synth = tts_stage.make_edge_synth(cfg["tts"])
            with ThreadPoolExecutor(max_workers=2) as pool:      # how run_tts really calls it
                ok = pool.submit(synth, "Hello.", "en-US-AriaNeural", "+0Hz", "+0%", "/x/s.mp3").result()
        self.assertTrue(ok)
        a = calls[0]
        self.assertEqual((a["text"], a["voice"], a["pitch"], a["rate"], a["out_path"]),
                         ("Hello.", "en-US-AriaNeural", "+0Hz", "+0%", "/x/s.mp3"))
        self.assertEqual((a["max_retries"], a["base_delay"]), (2, 0.5))
        self.assertEqual(a["label"], "s.mp3")                   # not the Vietnamese default label

    def test_edge_synth_failure_returns_false_instead_of_raising(self):
        async def boom(*a, **k):
            raise ConnectionError("no internet")

        with mock.patch.object(tts_stage.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(tts, "_synth_one", boom):
            synth = tts_stage.make_edge_synth(make_cfg()["tts"])
            self.assertFalse(synth("Hi.", "v", "+0Hz", "+0%", "/x/s.mp3"))

    def test_edge_synth_timeout_returns_false(self):
        async def hang(*a, **k):
            await asyncio.sleep(30)

        cfg = make_cfg(tts={"timeout": 0.05, "max_retries": 1})
        with mock.patch.object(tts_stage.importlib.util, "find_spec", return_value=object()), \
                mock.patch.object(tts, "_synth_one", hang):
            synth = tts_stage.make_edge_synth(cfg["tts"])
            self.assertFalse(synth("Hi.", "v", "+0Hz", "+0%", "/x/s.mp3"))


class LlmCallSiteTest(unittest.TestCase):
    def test_api_params_tuple_shape_matches_what_llm_translator_unpacks(self):
        key, model, base, timeout = translate.api_params_for_provider(
            {"gemini_api_key": "K"}, "gemini")
        self.assertEqual(key, "K")
        self.assertTrue(model)
        self.assertIsInstance(timeout, int)

    def test_llm_translator_calls_api_call_with_valid_arguments_and_real_parser(self):
        calls = []
        real_sig = inspect.signature(translate._api_call)

        def fake_call(*a, **k):
            b = real_sig.bind(*a, **k)
            b.apply_defaults()
            calls.append(b.arguments)
            return "```json\n[\"Hello everyone.\", \"Thanks!\"]\n```"    # fenced, as LLMs often do

        t = tl.LLMTranslator("gemini", {"gemini_api_key": "K"}, call=fake_call)
        out = t.translate_batch(["hola a todos", "gracias"], "es", budgets=[30, 12])
        self.assertEqual(out, ["Hello everyone.", "Thanks!"])
        a = calls[0]
        self.assertEqual((a["api_key"], a["provider"]), ("K", "gemini"))
        self.assertIn("max_chars", a["prompt"])

    def test_real_json_line_parser_handles_typical_llm_replies(self):
        parse = translate._parse_json_lines
        self.assertEqual(parse('["a", "b"]', 2), ["a", "b"])
        self.assertEqual(parse('```json\n["a", "b"]\n```', 2), ["a", "b"])
        self.assertIsNone(parse('["a"]', 2))                     # wrong count -> rejected
        self.assertIsNone(parse("no json here", 2))
        # The legacy parser rejects prose around the array (documented here on purpose)...
        self.assertIsNone(parse('Here you go:\n["a", "b"]', 2))

    def test_llm_translator_tolerates_prose_around_the_json_array(self):
        replies = iter(['Sure! Here is the translation:\n["Hello.", "Bye."]\nHope that helps!'])
        calls = []

        def fake_call(*a, **k):
            calls.append(1)
            return next(replies)

        t = tl.LLMTranslator("gemini", {"gemini_api_key": "K"}, call=fake_call)
        self.assertEqual(t.translate_batch(["hola", "adios"], "es"), ["Hello.", "Bye."])
        self.assertEqual(len(calls), 1)                          # accepted first time, no paid retry
        self.assertEqual(tl._extract_array("no brackets"), "no brackets")


if __name__ == "__main__":
    unittest.main()
