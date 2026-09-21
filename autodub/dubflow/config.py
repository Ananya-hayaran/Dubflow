"""Configuration for the English DubFlow pipeline.

Everything has a sensible default, so the pipeline runs with NO config file.
To change something, copy ``dubflow.example.yaml`` to ``dubflow.yaml`` and edit
only the lines you care about; missing keys fall back to DEFAULTS below.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

from ..utils import log

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_NAME = "dubflow.yaml"

DEFAULTS: Dict[str, Any] = {
    # Where downloads / intermediate files / outputs live (relative to project root).
    "data_dir": "data",
    # Folder containing ffmpeg/ffprobe if they are not on PATH ('' = use PATH).
    "ffmpeg_dir": "",
    "download": {
        "quality": "720",              # best | 1080 | 720 | 480 | 360
        "prefer_h264": True,           # lets the final render copy the video stream
        "concurrent_fragments": 8,
        "external_downloader": "auto",
        "cookies_from_browser": None,  # e.g. "chrome" for age-restricted videos
        "cookies_file": None,
    },
    "asr": {
        "backend": "faster-whisper",
        "model_size": "small",         # tiny | base | small | medium | large-v3
        "language": None,              # None = auto-detect
        "device": "auto",              # auto -> cuda if available else cpu
        "compute_type": "auto",        # auto -> float16 on cuda, int8 on cpu
        "beam_size": 5,
        "loudnorm": True,
        "rescue_gaps": True,           # re-transcribe long uncovered stretches
        "min_gap_seconds": 10,
        "max_rescue_rounds": 2,
        "silence_db": -45,
        "filter_hallucinations": True,
    },
    "translation": {
        # local        = free, offline model (NLLB-200), no API key
        # passthrough  = no translation (auto-used when the source is English)
        # gemini | nvidia | tokenrouter | inferx | zenmux = optional LLM APIs (need a key)
        "provider": "local",
        "local_model": "auto",         # auto = facebook/nllb-200-distilled-600M, or any HF model id
        "batch_size": 16,
        "chars_per_sec": 15.0,         # target speaking rate used to budget line length
        "force": False,                # translate even if the source is already English
        "api": {},                     # e.g. {gemini_api_key: "...", gemini_model: "..."}
    },
    "tts": {
        "engine": "edge",
        "voice": "en-US-AriaNeural",         # ONE consistent voice for the whole video
        "fallback_voice": "en-US-GuyNeural",  # only used if a line fails with the main voice
        "rate": "+0%",
        "pitch": "+0Hz",
        "concurrency": 6,
        "max_retries": 4,
        "retry_delay": 1.2,
        "timeout": 60,
        "trim_silence": True,
    },
    "alignment": {
        "min_speed": None,             # None -> alignment.MIN_TTS_SPEED (0.85)
        "max_speed": None,             # None -> alignment.MAX_TTS_SPEED (1.25)
        "tolerance": None,             # None -> alignment.ALIGNMENT_TOLERANCE (0.05)
        "max_overhang_seconds": 0.0,   # 0 = clip must fit its own [start, end] window
        "stretch_short_segments": False,  # slow very short clips down (never below min_speed)
    },
    "mix": {
        "sample_rate": 48000,
        "mode": "ffmpeg",              # ffmpeg = deterministic, no torch needed
        "chunk_seconds": 120,          # long videos are mixed in windows, not in RAM
    },
    "render": {
        "keep_original_db": None,      # None = English dub only; e.g. -25 keeps quiet original
        "force_h264": True,            # re-encode only if the source is not H.264
        "use_gpu": True,
        "x264_preset": "veryfast",
        "cpu_threads": 4,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in out:
            log(f"Config: ignoring unknown key '{path}{key}'.", "warn")
            continue
        if isinstance(out[key], dict) and isinstance(value, dict) and out[key]:
            out[key] = _deep_merge(out[key], value, f"{path}{key}.")
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load DEFAULTS, merged with ``path`` (or ./dubflow.yaml when it exists)."""
    if path is None:
        candidate = os.path.join(PROJECT_ROOT, DEFAULT_CONFIG_NAME)
        path = candidate if os.path.isfile(candidate) else None
    if not path:
        return copy.deepcopy(DEFAULTS)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("PyYAML is required to read a config file: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as fh:
        try:
            user = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(user, dict):
        raise ValueError(f"{path} must contain a mapping of settings at the top level.")
    return _deep_merge(DEFAULTS, user)


def detect_device() -> Tuple[str, str]:
    """(device, compute_type) for faster-whisper: CUDA+float16 if usable, else CPU+int8."""
    try:
        import ctranslate2  # installed together with faster-whisper

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def resolve_asr_device(asr_cfg: Dict[str, Any]) -> Tuple[str, str]:
    device = str(asr_cfg.get("device") or "auto").lower()
    ctype = str(asr_cfg.get("compute_type") or "auto").lower()
    if device == "auto" or ctype == "auto":
        auto_dev, auto_ctype = detect_device()
        if device == "auto":
            device = auto_dev
        if ctype == "auto":
            # Keep compute type consistent with the *chosen* device.
            ctype = "float16" if device == "cuda" else "int8"
    return device, ctype
