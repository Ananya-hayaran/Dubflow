"""Cache/validity checks. A file existing is NOT proof that it is usable."""
from __future__ import annotations

import hashlib
import json
import os
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils import run
from .errors import ValidationError


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def sha_short(*parts: Any, length: int = 12) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON so readers never see a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str, required_keys: Iterable[str] = ()) -> Tuple[bool, Optional[Any]]:
    """(ok, data). ok is False when missing, empty, unparsable or lacking keys."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False, None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False, None
    if required_keys:
        if not isinstance(data, dict) or any(k not in data for k in required_keys):
            return False, None
    return True, data


# --------------------------------------------------------------------------- #
#  media probing
# --------------------------------------------------------------------------- #
@dataclass
class MediaInfo:
    size: int = 0
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    pix_fmt: str = ""
    width: int = 0
    height: int = 0
    video_duration: float = 0.0
    audio_duration: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe_media(path: str) -> Optional[MediaInfo]:
    """ffprobe summary of a media file, or None if it cannot be opened."""
    if not path or not os.path.isfile(path):
        return None
    try:
        res = run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                   "-of", "json", path], check=False, timeout=120)
        data = json.loads(res.stdout or "{}")
    except Exception:
        return None
    if not data.get("format") and not data.get("streams"):
        return None
    info = MediaInfo(size=os.path.getsize(path),
                     duration=_f((data.get("format") or {}).get("duration")))
    for st in data.get("streams") or []:
        kind = st.get("codec_type")
        if kind == "video" and not info.has_video:
            # Cover art / thumbnails show up as video streams with 0 real frames.
            if st.get("disposition", {}).get("attached_pic"):
                continue
            info.has_video = True
            info.video_codec = st.get("codec_name", "")
            info.pix_fmt = st.get("pix_fmt", "")
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.video_duration = _f(st.get("duration")) or info.duration
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = st.get("codec_name", "")
            info.audio_duration = _f(st.get("duration")) or info.duration
    return info


def wav_duration(path: str) -> Optional[float]:
    """Duration of a PCM WAV read with the stdlib (no subprocess), or None."""
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, EOFError, OSError):
        return None


def audio_duration(path: str) -> float:
    """Duration in seconds, 0.0 if the file is missing/corrupt. Fast for WAV."""
    if not path or not os.path.isfile(path):
        return 0.0
    d = wav_duration(path)
    if d is not None:
        return d
    info = probe_media(path)
    return info.duration if info and info.has_audio else 0.0


def audio_file_ok(path: str, min_duration: float = 0.02,
                  expect_duration: Optional[float] = None, tol: float = 0.05) -> bool:
    """Audio can be opened, is non-trivial, and (optionally) has the expected length."""
    d = audio_duration(path)
    if d < min_duration:
        return False
    if expect_duration is not None and abs(d - expect_duration) > max(tol, expect_duration * 0.02):
        return False
    return True


def source_video_ok(path: str) -> bool:
    """A downloaded/local video is usable: opens, has video AND audio, has length."""
    info = probe_media(path)
    return bool(info and info.has_video and info.has_audio and info.duration > 0.1)


# --------------------------------------------------------------------------- #
#  final MP4
# --------------------------------------------------------------------------- #
def validate_final_mp4(path: str, source_duration: float,
                       av_tolerance: float = 0.5,
                       duration_tolerance: float = 1.0) -> Dict[str, Any]:
    """Check the finished file with ffprobe. Raises ValidationError on any failure.

    Checks: exists, size > 0, video stream, audio stream, duration close to the
    source video, audio/video durations compatible. Returns a report dict.
    """
    problems: List[str] = []
    if not os.path.isfile(path):
        raise ValidationError(f"Final video does not exist: {path}")
    size = os.path.getsize(path)
    if size <= 0:
        raise ValidationError(f"Final video is empty (0 bytes): {path}")
    info = probe_media(path)
    if info is None:
        raise ValidationError(f"ffprobe cannot open the final video: {path}")
    if not info.has_video:
        problems.append("no video stream")
    if not info.has_audio:
        problems.append("no audio stream")
    # tolerances scale a little for long videos
    dur_tol = max(duration_tolerance, source_duration * 0.02)
    av_tol = max(av_tolerance, info.duration * 0.02)
    if source_duration > 0 and abs(info.duration - source_duration) > dur_tol:
        problems.append(f"duration {info.duration:.2f}s differs from source "
                        f"{source_duration:.2f}s by more than {dur_tol:.2f}s")
    if info.has_video and info.has_audio and \
            abs(info.video_duration - info.audio_duration) > av_tol:
        problems.append(f"audio ({info.audio_duration:.2f}s) and video "
                        f"({info.video_duration:.2f}s) durations differ by more than {av_tol:.2f}s")
    report = info.as_dict()
    report.update({"path": path, "source_duration": round(source_duration, 3),
                   "checks_passed": not problems, "problems": problems})
    if problems:
        raise ValidationError("Final video failed validation: " + "; ".join(problems))
    return report


def measure_max_volume_db(path: str) -> Optional[float]:
    """Peak level of an audio file via ffmpeg volumedetect (None if unknown)."""
    try:
        res = run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "volumedetect",
                   "-vn", "-f", "null", "-"], check=False, timeout=3600)
        text = (res.stderr or "") + (res.stdout or "")
        for line in text.splitlines():
            if "max_volume:" in line:
                return float(line.split("max_volume:")[1].split("dB")[0].strip())
    except Exception:
        return None
    return None
