"""Shared test helpers: synthetic media (real ffmpeg) and fake ASR/translation/TTS engines.

Nothing here touches the network or needs a model download.
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autodub.dubflow.config import DEFAULTS  # noqa: E402
from autodub.dubflow.translation import Translator  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True,
                   stdin=subprocess.DEVNULL)


def make_video(path: str, seconds: float = 12.0, size: str = "320x180",
               audio: bool = True, video: bool = True) -> str:
    """Tiny H.264/AAC MP4 with a moving test pattern and a quiet tone."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    args = []
    if video:
        args += ["-f", "lavfi", "-i", f"testsrc2=s={size}:r=15:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=220:d={seconds}:sample_rate=44100"]
    if video:
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k"]
    ff(*args, "-shortest" if audio and video else "-t", *(() if audio and video else (str(seconds),)), path)
    return path


def make_tone(path: str, seconds: float, freq: int = 440, fmt: str = "wav",
              lead_silence: float = 0.0, tail_silence: float = 0.0) -> str:
    """Write a sine tone (optionally padded with silence) to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    af = []
    if lead_silence > 0:
        ms = int(lead_silence * 1000)
        af.append(f"adelay={ms}|{ms}")
    if tail_silence > 0:
        af.append(f"apad=pad_dur={tail_silence}")
    args = ["-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}:sample_rate=24000"]
    if af:
        args += ["-af", ",".join(af)]
    if fmt == "wav":
        args += ["-ac", "1", "-c:a", "pcm_s16le"]
    ff(*args, "-f", fmt, path)
    return path


def make_cfg(**overrides: Any) -> Dict[str, Any]:
    """Default config with fast test settings. Pass e.g. tts={'concurrency': 2}."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg["render"]["use_gpu"] = False
    cfg["tts"]["concurrency"] = 3
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


# ---------------------------------------------------------------------------
#  fake engines
# ---------------------------------------------------------------------------
class FakeASR:
    """Callable transcriber returning fixed segments; counts calls."""

    def __init__(self, segments: Sequence[Sequence[Any]], language: str = "es"):
        self.segments = segments
        self.language = language
        self.calls = 0

    def __call__(self, audio_path: str, cfg: Dict[str, Any]):
        self.calls += 1
        return [SimpleNamespace(start=s, end=e, text=t) for s, e, t in self.segments], self.language


class FakeTranslator(Translator):
    """Deterministic 'translation': prefixes text, optionally lengthens it."""
    name = "fake"

    def __init__(self, prefix: str = "EN: ", repeat: Optional[Dict[str, int]] = None,
                 blank_for: Sequence[str] = ()):
        self.prefix, self.repeat, self.blank_for = prefix, repeat or {}, set(blank_for)
        self.calls: List[List[str]] = []

    def translate_batch(self, texts, source_lang, budgets=None):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            if t in self.blank_for:
                out.append("")
            else:
                out.append((self.prefix + t) + (" and more words" * self.repeat.get(t, 0)))
        return out


class FakeSynth:
    """TTS stand-in: writes a real sine 'voice' whose length grows with the text."""

    def __init__(self, seconds_per_char: float = 0.05, base: float = 0.2,
                 fail_texts: Sequence[str] = (), fail_main_voice_only: bool = False,
                 durations: Optional[Dict[str, float]] = None):
        self.spc, self.base = seconds_per_char, base
        self.fail_texts = set(fail_texts)
        self.fail_main_voice_only = fail_main_voice_only
        self.durations = durations or {}
        self.calls: List[Dict[str, str]] = []

    def __call__(self, text: str, voice: str, pitch: str, rate: str, out_path: str) -> bool:
        self.calls.append({"text": text, "voice": voice, "out": out_path})
        if text in self.fail_texts:
            if not self.fail_main_voice_only or voice == "en-US-AriaNeural":
                return False
        dur = self.durations.get(text, self.base + self.spc * len(text))
        make_tone(out_path, dur, fmt="wav", lead_silence=0.15, tail_silence=0.15)
        return True


class TempDirCase(unittest.TestCase):
    """Base class giving every test its own throw-away data directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="dubflow_test_")
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        from autodub import utils
        self.addCleanup(utils.stop_file_log)
        if not os.environ.get("DUBFLOW_TEST_VERBOSE"):       # keep test output readable
            real = utils._safe_print
            utils._safe_print = lambda line: None
            self.addCleanup(setattr, utils, "_safe_print", real)
