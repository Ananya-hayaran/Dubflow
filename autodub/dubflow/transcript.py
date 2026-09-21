"""Stage: speech-to-text -> ``transcript.json`` (timestamped, id-keyed).

Reuses the existing ``autodub.asr.transcribe`` (faster-whisper backend with
temperature fallback, VAD, gap rescue and hallucination filtering).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils import log
from .config import resolve_asr_device
from .errors import PipelineError
from .validate import atomic_write_json, read_json
from .workspace import Workspace

TRANSCRIPT_VERSION = 1
MIN_SEGMENT_SECONDS = 0.05

# A transcriber takes (audio_path, cfg) and returns (segments, language) where
# segments have .start/.end/.text (asr.Segment or any duck-typed object).
Transcriber = Callable[[str, Dict[str, Any]], Tuple[List[Any], str]]


def _default_transcriber(audio_path: str, cfg: Dict[str, Any]) -> Tuple[List[Any], str]:
    from .. import asr  # heavy module; imported only when actually transcribing

    a = cfg["asr"]
    device, compute_type = resolve_asr_device(a)
    log(f"ASR: faster-whisper '{a['model_size']}' on {device}/{compute_type}", "info")
    return asr.transcribe(
        audio_path, backend=a["backend"], language=a.get("language"),
        model_size=a["model_size"], device=device, compute_type=compute_type,
        beam_size=int(a.get("beam_size", 5)),
        rescue_gaps=bool(a.get("rescue_gaps", True)),
        min_gap_seconds=float(a.get("min_gap_seconds", 10)),
        max_rescue_rounds=int(a.get("max_rescue_rounds", 2)),
        silence_db=float(a.get("silence_db", -45)),
        fallback_backend=None,                       # no Chinese-only fallbacks
        filter_hallucinations=bool(a.get("filter_hallucinations", True)),
    )


def _is_speakable(text: str) -> bool:
    return any(c.isalnum() for c in (text or ""))


def build_segments(raw: List[Any], duration: float) -> List[Dict[str, Any]]:
    """Clean ASR output into id-keyed segment dicts (ids are 1..N in time order)."""
    rows = []
    for s in raw:
        text = " ".join(str(getattr(s, "text", "")).split())
        if not _is_speakable(text):
            continue
        start = max(0.0, float(s.start))
        end = float(s.end)
        if duration > 0:
            if start >= duration:
                continue
            end = min(end, duration)
        if end < start + MIN_SEGMENT_SECONDS:
            end = start + MIN_SEGMENT_SECONDS
            if duration > 0:
                end = min(end, duration)
            if end <= start:
                continue
        rows.append((start, end, text))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [{"id": i, "start": round(st, 3), "end": round(en, 3), "text": tx}
            for i, (st, en, tx) in enumerate(rows, 1)]


def validate_transcript(data: Any, audio_duration: float) -> bool:
    """Structural + consistency check of a transcript.json payload."""
    if not isinstance(data, dict) or data.get("version") != TRANSCRIPT_VERSION:
        return False
    segs = data.get("segments")
    if not isinstance(segs, list) or not segs:
        return False
    try:
        stored = float(data.get("audio_duration", 0))
        if audio_duration > 0 and abs(stored - audio_duration) > 1.0:
            return False                      # transcript belongs to different audio
        for expect_id, s in enumerate(segs, 1):
            if int(s["id"]) != expect_id:
                return False
            if not (0 <= float(s["start"]) < float(s["end"])):
                return False
            if not str(s["text"]).strip():
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return bool(str(data.get("source_language", "")).strip())


def run_transcription(ws: Workspace, cfg: Dict[str, Any], audio_path: str,
                      audio_duration: float, transcriber: Optional[Transcriber] = None,
                      force: bool = False) -> Tuple[Dict[str, Any], bool]:
    """Return (transcript, cache_hit)."""
    if not force:
        ok, data = read_json(ws.transcript_path, ("segments", "source_language"))
        if ok and validate_transcript(data, audio_duration):
            log(f"Transcript cache hit: {len(data['segments'])} segments, "
                f"language={data['source_language']}", "ok")
            return data, True
        if ok:
            log("Transcript cache is invalid or belongs to different audio; re-transcribing.", "warn")
    log("Transcript cache miss: running speech recognition...", "info")

    t0 = time.time()
    raw, language = (transcriber or _default_transcriber)(audio_path, cfg)
    segments = build_segments(raw, audio_duration)
    if not segments:
        raise PipelineError(
            "No speech was detected in this video, so there is nothing to dub. "
            "If the video does contain speech, try a larger model (asr.model_size: medium).")
    language = (str(language or "").strip().lower() or "unknown")
    a = cfg["asr"]
    payload = {
        "version": TRANSCRIPT_VERSION,
        "video_id": ws.video_id,
        "source_language": language,
        "audio_duration": round(float(audio_duration), 3),
        "asr": {"backend": a["backend"], "model_size": a["model_size"]},
        "segments": segments,
    }
    atomic_write_json(ws.transcript_path, payload)
    log(f"Transcribed {len(segments)} segments in {time.time() - t0:.1f}s "
        f"(language={language}).", "ok")
    return payload, False
