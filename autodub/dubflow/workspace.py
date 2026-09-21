"""On-disk layout for one video, keyed by a stable ``video_id``.

    data/
      intermediate/<video_id>/
          source/                  downloaded video (yt-dlp output)
          audio_16k.wav            audio extracted for speech recognition
          transcript.json          timestamped transcript (segment ids)
          translations.json        English translation, same ids/timestamps
          tts/segment_0001.wav     one English clip per segment (+ .json metadata)
          aligned/segment_0001.wav clips after timing alignment (+ .json metadata)
          alignment.json           per-segment timing report (needs_review, speed...)
          dubbed_audio.wav         full-length English dub track (+ .meta.json)
          pipeline.log             everything logged during runs
      output/
          <video_id>_dubbed.mp4    final video
          <video_id>_en.srt        English subtitles
          <video_id>_metrics.json  metrics and stage/caching report
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return slug[:limit] or "video"


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", (text or "").strip(), re.I))


def derive_video_id(source: str) -> str:
    """Stable id for a YouTube URL, any other URL, or a local file path.

    * YouTube (watch?v=, youtu.be/, /shorts/, /embed/, /live/) -> the 11-char id.
    * Other URLs -> ``<host-slug>-<sha1[:8]>``.
    * Local files -> ``<name-slug>-<sha1(abs path)[:8]>`` (same file, same id).
    """
    source = (source or "").strip().strip('"').strip("'")
    if is_url(source):
        parsed = urlparse(source)
        host = (parsed.hostname or "").lower()
        candidate = None
        if host.endswith("youtu.be"):
            candidate = parsed.path.strip("/").split("/")[0]
        elif "youtube" in host:
            qs = parse_qs(parsed.query)
            if qs.get("v"):
                candidate = qs["v"][0]
            else:
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                    candidate = parts[1]
        if candidate and _YT_ID.match(candidate):
            return candidate
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return f"{_slug(host.replace('www.', ''), 20)}-{digest}"
    path = os.path.abspath(source)
    stem = os.path.splitext(os.path.basename(path))[0]
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(stem)}-{digest}"


@dataclass
class Workspace:
    data_dir: str
    video_id: str

    # ---- intermediate ----
    @property
    def root(self) -> str:
        return os.path.join(self.data_dir, "intermediate", self.video_id)

    @property
    def source_dir(self) -> str:
        return os.path.join(self.root, "source")

    @property
    def audio_path(self) -> str:
        return os.path.join(self.root, "audio_16k.wav")

    @property
    def transcript_path(self) -> str:
        return os.path.join(self.root, "transcript.json")

    @property
    def translations_path(self) -> str:
        return os.path.join(self.root, "translations.json")

    @property
    def tts_dir(self) -> str:
        return os.path.join(self.root, "tts")

    @property
    def tts_raw_dir(self) -> str:
        return os.path.join(self.root, "tts", "_raw")

    @property
    def aligned_dir(self) -> str:
        return os.path.join(self.root, "aligned")

    @property
    def alignment_path(self) -> str:
        return os.path.join(self.root, "alignment.json")

    @property
    def dubbed_audio_path(self) -> str:
        return os.path.join(self.root, "dubbed_audio.wav")

    @property
    def dubbed_meta_path(self) -> str:
        return os.path.join(self.root, "dubbed_audio.meta.json")

    @property
    def final_meta_path(self) -> str:
        return os.path.join(self.root, "final_video.meta.json")

    @property
    def log_path(self) -> str:
        return os.path.join(self.root, "pipeline.log")

    # ---- output ----
    @property
    def output_dir(self) -> str:
        return os.path.join(self.data_dir, "output")

    @property
    def final_mp4(self) -> str:
        return os.path.join(self.output_dir, f"{self.video_id}_dubbed.mp4")

    @property
    def partial_mp4(self) -> str:
        # Rendered here first and renamed only after validation, so a crash can
        # never leave a half-written file that looks like a finished result.
        return os.path.join(self.output_dir, f"{self.video_id}_dubbed.partial.mp4")

    @property
    def srt_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.video_id}_en.srt")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.video_id}_metrics.json")

    # ---- per-segment files ----
    def tts_wav(self, seg_id: int) -> str:
        return os.path.join(self.tts_dir, f"segment_{seg_id:04d}.wav")

    def aligned_wav(self, seg_id: int) -> str:
        return os.path.join(self.aligned_dir, f"segment_{seg_id:04d}.wav")

    def ensure(self) -> "Workspace":
        for d in (self.source_dir, self.tts_dir, self.tts_raw_dir,
                  self.aligned_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)
        return self
