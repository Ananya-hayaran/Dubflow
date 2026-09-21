"""2. Translation output contract, caching and partial resume (no paid APIs, no model downloads)."""
import json
import os
import unittest

from tests.helpers import FakeTranslator, TempDirCase, make_cfg

from autodub.dubflow import translation as tr
from autodub.dubflow.errors import TranslationError
from autodub.dubflow.validate import read_json
from autodub.dubflow.workspace import Workspace

SEGS = [
    {"id": 1, "start": 0.5, "end": 2.0, "text": "hola a todos"},
    {"id": 2, "start": 2.4, "end": 4.75, "text": "hoy aprendemos algo nuevo"},
    {"id": 3, "start": 5.0, "end": 6.125, "text": "gracias"},
]


def transcript(segs=SEGS, lang="es"):
    return {"version": 1, "source_language": lang, "audio_duration": 10.0,
            "segments": [dict(s) for s in segs]}


class ContractTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.tmp, "vid").ensure()
        self.cfg = make_cfg()

    def test_ids_timestamps_and_source_text_are_preserved(self):
        recs, info = tr.run_translation(self.ws, self.cfg, transcript(), FakeTranslator())
        self.assertEqual(len(recs), 3)
        for seg, rec in zip(SEGS, recs):
            self.assertEqual(rec["id"], seg["id"])
            self.assertEqual(rec["start"], seg["start"])          # exactly equal, not "close"
            self.assertEqual(rec["end"], seg["end"])
            self.assertEqual(rec["source_text"], seg["text"])
            self.assertEqual(rec["translated_text"], "EN: " + seg["text"])
        self.assertEqual(info["translated_now"], 3)

    def test_json_file_written_with_english_target(self):
        tr.run_translation(self.ws, self.cfg, transcript(), FakeTranslator())
        ok, data = read_json(self.ws.translations_path, ("segments", "target_language"))
        self.assertTrue(ok)
        self.assertEqual(data["target_language"], "en")
        self.assertTrue(data["complete"])
        self.assertEqual(data["segments"][1]["end"], 4.75)

    def test_second_run_uses_cache_and_never_calls_translator(self):
        tr.run_translation(self.ws, self.cfg, transcript(), FakeTranslator())
        second = FakeTranslator()
        recs, info = tr.run_translation(self.ws, self.cfg, transcript(), second)
        self.assertEqual(second.calls, [])
        self.assertEqual(info["cache_hits"], 3)
        self.assertEqual(info["translated_now"], 0)
        self.assertEqual(recs[0]["translated_text"], "EN: hola a todos")

    def test_changed_source_text_only_retranslates_that_segment(self):
        tr.run_translation(self.ws, self.cfg, transcript(), FakeTranslator())
        changed = [dict(s) for s in SEGS]
        changed[1]["text"] = "texto distinto"
        fake = FakeTranslator()
        recs, info = tr.run_translation(self.ws, self.cfg, transcript(changed), fake)
        self.assertEqual(fake.calls, [["texto distinto"]])
        self.assertEqual(info["cache_hits"], 2)
        self.assertEqual(recs[1]["translated_text"], "EN: texto distinto")

    def test_changed_timestamps_invalidate_cached_translation(self):
        tr.run_translation(self.ws, self.cfg, transcript(), FakeTranslator())
        moved = [dict(s) for s in SEGS]
        moved[0]["start"] = 0.75
        fake = FakeTranslator()
        tr.run_translation(self.ws, self.cfg, transcript(moved), fake)
        self.assertEqual(fake.calls, [["hola a todos"]])

    def test_corrupt_translation_file_is_ignored_not_trusted(self):
        with open(self.ws.translations_path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        fake = FakeTranslator()
        recs, info = tr.run_translation(self.ws, self.cfg, transcript(), fake)
        self.assertEqual(info["cache_hits"], 0)
        self.assertEqual(len(recs), 3)

    def test_partial_progress_survives_a_crash_and_resumes(self):
        cfg = make_cfg(translation={"batch_size": 1})

        class Flaky(FakeTranslator):
            def translate_batch(self, texts, source_lang, budgets=None):
                if len(self.calls) == 1:
                    raise RuntimeError("network died")
                return super().translate_batch(texts, source_lang, budgets)

        with self.assertRaises(RuntimeError):
            tr.run_translation(self.ws, cfg, transcript(), Flaky())
        ok, data = read_json(self.ws.translations_path, ("segments",))
        self.assertTrue(ok)
        self.assertFalse(data["complete"])
        self.assertEqual([s["id"] for s in data["segments"]], [1])   # batch 1 was saved
        resumed = FakeTranslator()
        recs, info = tr.run_translation(self.ws, cfg, transcript(), resumed)
        self.assertEqual(resumed.calls, [["hoy aprendemos algo nuevo"], ["gracias"]])
        self.assertEqual(info["cache_hits"], 1)
        self.assertEqual(len(recs), 3)

    def test_empty_translation_is_flagged_not_silently_dropped(self):
        recs, _ = tr.run_translation(self.ws, self.cfg, transcript(),
                                     FakeTranslator(blank_for=["gracias"]))
        self.assertEqual(len(recs), 3)                     # segment still present
        self.assertEqual(recs[2]["translated_text"], "")
        self.assertEqual(recs[2]["flag"], "empty_translation")

    def test_wrong_line_count_from_translator_is_an_error(self):
        class Short(FakeTranslator):
            def translate_batch(self, texts, source_lang, budgets=None):
                return ["only one"]
        with self.assertRaises(TranslationError):
            tr.run_translation(self.ws, self.cfg, transcript(), Short())

    def test_reading_pressure_flags_lines_too_long_for_window(self):
        recs = [{"id": 1, "start": 0.0, "end": 1.0, "translated_text": "x" * 60},
                {"id": 2, "start": 2.0, "end": 6.0, "translated_text": "short"}]
        rp = tr.reading_pressure(recs, 15.0)
        self.assertEqual(rp["likely_too_long"], 1)
        self.assertEqual(rp["ids"], [1])


class ProviderSelectionTest(unittest.TestCase):
    def test_english_source_is_passthrough(self):
        t = tr.make_translator(make_cfg(), "en")
        self.assertIsInstance(t, tr.PassthroughTranslator)
        self.assertEqual(t.translate_batch([" Hello   world "], "en"), ["Hello world"])

    def test_non_english_uses_free_local_model_by_default(self):
        t = tr.make_translator(make_cfg(), "es")
        self.assertIsInstance(t, tr.LocalTranslator)
        self.assertEqual(t.model_id, "facebook/nllb-200-distilled-600M")

    def test_force_translates_even_english(self):
        cfg = make_cfg(translation={"force": True})
        self.assertIsInstance(tr.make_translator(cfg, "en"), tr.LocalTranslator)

    def test_browser_and_unknown_providers_rejected_clearly(self):
        with self.assertRaises(TranslationError):
            tr.make_translator(make_cfg(translation={"provider": "browser"}), "es")
        with self.assertRaises(TranslationError):
            tr.make_translator(make_cfg(translation={"provider": "nope"}), "es")

    def test_llm_provider_without_key_fails_with_actionable_message(self):
        for var in ("GEMINI_API_KEY",):
            os.environ.pop(var, None)
        t = tr.make_translator(make_cfg(translation={"provider": "gemini"}), "es")
        with self.assertRaises(TranslationError) as cm:
            t.translate_batch(["hola"], "es")
        self.assertIn("API key", str(cm.exception))
        self.assertIn("provider: local", str(cm.exception))

    def test_nllb_language_codes(self):
        self.assertEqual(tr.to_nllb_code("es"), "spa_Latn")
        self.assertEqual(tr.to_nllb_code("hi"), "hin_Deva")
        self.assertEqual(tr.to_nllb_code("zh-CN"), "zho_Hans")
        with self.assertRaises(TranslationError):
            tr.to_nllb_code("xx")


class LocalTranslatorLogicTest(unittest.TestCase):
    """The model itself is not downloaded; batching/ordering logic is tested with a stub."""

    def test_batches_in_order_and_cleans_output(self):
        seen = []

        class Stub(tr.LocalTranslator):
            def _translate_chunk(self, chunk, source_lang):
                seen.append(list(chunk))
                return [f"  T({c}) " for c in chunk]

        t = Stub(batch_size=2)
        out = t.translate_batch(["a", "b", "c", "d", "e"], "es")
        self.assertEqual(seen, [["a", "b"], ["c", "d"], ["e"]])
        self.assertEqual(out, ["T(a)", "T(b)", "T(c)", "T(d)", "T(e)"])

    def test_unsupported_language_fails_fast_before_loading_model(self):
        t = tr.LocalTranslator()
        with self.assertRaises(TranslationError):
            t.translate_batch(["x"], "xx")

    def test_missing_torch_gives_install_hint(self):
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed here")
        with self.assertRaises(TranslationError) as cm:
            tr.LocalTranslator().translate_batch(["hola"], "es")
        self.assertIn("pip install", str(cm.exception))


class LLMTranslatorTest(unittest.TestCase):
    def test_prompt_asks_for_natural_concise_spoken_english_and_parses_json(self):
        prompts = []

        def fake_call(prompt, key, model, temp, provider="gemini", api_base_url=None, api_timeout=1):
            prompts.append(prompt)
            return json.dumps(["Hello everyone.", "Thanks!"])

        t = tr.LLMTranslator("gemini", {"gemini_api_key": "k"}, call=fake_call)
        out = t.translate_batch(["hola a todos", "gracias"], "es", budgets=[30, 12])
        self.assertEqual(out, ["Hello everyone.", "Thanks!"])
        p = prompts[0].lower()
        self.assertIn("english", p)
        self.assertIn("spoken aloud", p)
        self.assertIn("concise", p)
        self.assertIn("max_chars", p)

    def test_retries_then_fails_on_unparseable_reply(self):
        calls = []

        def bad(prompt, *a, **k):
            calls.append(1)
            return "sorry I cannot"

        import time
        real_sleep, time.sleep = time.sleep, lambda s: None
        try:
            t = tr.LLMTranslator("gemini", {"gemini_api_key": "k"}, call=bad, retries=3)
            with self.assertRaises(TranslationError):
                t.translate_batch(["hola"], "es")
        finally:
            time.sleep = real_sleep
        self.assertEqual(len(calls), 3)


class CleanTest(unittest.TestCase):
    def test_clean_english(self):
        self.assertEqual(tr.clean_english('  "Hello\n  there"  '), "Hello there")
        self.assertEqual(tr.clean_english("a\u200bb"), "ab")
        self.assertEqual(tr.clean_english(None), "")

    def test_char_budget_scales_with_duration(self):
        self.assertGreater(tr.char_budget(0, 4, 15), tr.char_budget(0, 1, 15))
        self.assertGreaterEqual(tr.char_budget(0, 0.1, 15), 12)


if __name__ == "__main__":
    unittest.main()
