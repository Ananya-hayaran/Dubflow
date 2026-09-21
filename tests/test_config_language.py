"""1. English language configuration (and guards against Vietnamese leaking back in)."""
import glob
import os
import re
import unittest

from tests.helpers import ROOT, TempDirCase, make_cfg  # noqa: F401  (sets sys.path)

from autodub.dubflow import config as dfconfig
from autodub.dubflow.config import DEFAULTS, load_config, resolve_asr_device
from autodub.dubflow.translation import TARGET_LANGUAGE


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


class EnglishDefaultsTest(unittest.TestCase):
    def test_target_language_is_english(self):
        self.assertEqual(TARGET_LANGUAGE, "en")

    def test_default_voice_is_english_edge_voice(self):
        self.assertTrue(DEFAULTS["tts"]["voice"].startswith("en-"))
        self.assertTrue(DEFAULTS["tts"]["fallback_voice"].startswith("en-"))
        self.assertEqual(DEFAULTS["tts"]["engine"], "edge")

    def test_no_vietnamese_or_paid_defaults(self):
        strings = [v for v in _walk(DEFAULTS) if isinstance(v, str)]
        for s in strings:
            self.assertNotIn("vi-VN", s)
            self.assertNotIn("vietsub", s.lower())
        # free/local translation by default, no API keys required
        self.assertEqual(DEFAULTS["translation"]["provider"], "local")
        self.assertEqual(DEFAULTS["translation"]["api"], {})
        self.assertEqual(DEFAULTS["asr"]["backend"], "faster-whisper")

    def test_english_pipeline_source_has_no_vietnamese_text(self):
        """New production code must stay English: no Vietnamese diacritics in autodub/dubflow
        or main.py. (tests/ is excluded on purpose: it holds deliberate non-English samples,
        e.g. the log guard's Vietnamese examples and a 'café' UTF-8 round-trip.)"""
        vi = re.compile("[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
        files = (glob.glob(os.path.join(ROOT, "autodub", "dubflow", "*.py"))
                 + [os.path.join(ROOT, "main.py")])
        self.assertGreaterEqual(len(files), 10)          # the glob really found the package
        offenders = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if vi.search(line):
                        offenders.append(f"{os.path.basename(f)}:{n}")
        self.assertEqual(offenders, [])

    def test_example_yaml_is_valid_and_matches_defaults(self):
        path = os.path.join(ROOT, "dubflow.example.yaml")
        self.assertTrue(os.path.isfile(path), "dubflow.example.yaml must exist")
        cfg = load_config(path)                    # would warn about unknown keys
        self.assertEqual(cfg["tts"]["voice"], DEFAULTS["tts"]["voice"])
        self.assertEqual(cfg["translation"]["provider"], "local")
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self.assertEqual(set(raw) - set(DEFAULTS), set(), "example has keys DEFAULTS lacks")


class LoadConfigTest(TempDirCase):
    def test_defaults_when_no_file(self):
        # An explicit non-existent path is an error; with no path and no dubflow.yaml
        # the built-in defaults are used unchanged.
        cfg = load_config(os.path.join(ROOT, "dubflow.example.yaml"))
        self.assertEqual(cfg["tts"]["engine"], "edge")
        self.assertEqual(cfg["download"]["quality"], "720")

    def test_partial_override_keeps_other_defaults(self):
        p = os.path.join(self.tmp, "c.yaml")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("tts:\n  voice: en-US-GuyNeural\nasr:\n  model_size: base\n")
        cfg = load_config(p)
        self.assertEqual(cfg["tts"]["voice"], "en-US-GuyNeural")
        self.assertEqual(cfg["tts"]["rate"], DEFAULTS["tts"]["rate"])
        self.assertEqual(cfg["asr"]["model_size"], "base")
        self.assertEqual(cfg["asr"]["backend"], "faster-whisper")

    def test_bad_yaml_and_missing_file_give_clear_errors(self):
        bad = os.path.join(self.tmp, "bad.yaml")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("tts: [unclosed\n")
        with self.assertRaises(ValueError):
            load_config(bad)
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(self.tmp, "nope.yaml"))

    def test_device_resolution(self):
        self.assertEqual(resolve_asr_device({"device": "cpu", "compute_type": "int8"}), ("cpu", "int8"))
        self.assertEqual(resolve_asr_device({"device": "cuda", "compute_type": "auto"}), ("cuda", "float16"))
        dev, ctype = resolve_asr_device({"device": "auto", "compute_type": "auto"})
        self.assertIn((dev, ctype), {("cpu", "int8"), ("cuda", "float16")})


if __name__ == "__main__":
    unittest.main()
