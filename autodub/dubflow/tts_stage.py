"""Stage: English text-to-speech, one cached WAV per segment.

    data/intermediate/<video_id>/tts/segment_0001.wav   (+ segment_0001.json)

* One consistent voice for the whole video (``tts.voice``); no voice cloning.
* Each clip's sidecar JSON stores a hash of (text, voice, rate, pitch, trim). A clip
  is reused only if the hash matches AND the WAV opens with the recorded duration;
  anything else is regenerated. Only missing/invalid clips are synthesised.
* Reuses the existing edge-tts retry logic (``autodub.tts._synth_one``).
* A segment that cannot be synthesised is NEVER dropped silently: it is logged,
  counted, and returned with ``failed=True`` so alignment/metrics flag it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils import log, run
from .errors import PipelineError
from .validate import atomic_write_json, audio_duration, read_json, sha_short
from .workspace import Workspace

MIN_CLIP_SECONDS = 0.05
ENGINE = "edge"

# (text, voice, pitch, rate, out_path) -> True if a non-empty audio file was written.
SynthFn = Callable[[str, str, str, str, str], bool]


@dataclass
class TtsResult:
    id: int
    path: Optional[str]
    duration: float = 0.0
    cached: bool = False
    failed: bool = False
    skipped: bool = False
    voice_used: str = ""
    reason: str = ""


def make_edge_synth(tts_cfg: Dict[str, Any]) -> SynthFn:
    """edge-tts synthesiser with retries (reuses ``autodub.tts._synth_one``)."""
    if importlib.util.find_spec("edge_tts") is None:
        raise PipelineError(
            "The 'edge-tts' package is not installed. Run:  pip install edge-tts")
    from .. import tts as legacy_tts

    retries = int(tts_cfg.get("max_retries", 4))
    delay = float(tts_cfg.get("retry_delay", 1.2))
    timeout = float(tts_cfg.get("timeout", 60)) * max(1, retries)

    def synth(text: str, voice: str, pitch: str, rate: str, out_path: str) -> bool:
        async def _go() -> bool:
            return await asyncio.wait_for(
                legacy_tts._synth_one(text, voice, pitch, rate, out_path,
                                      max_retries=retries, base_delay=delay,
                                      label=os.path.basename(out_path)),
                timeout=timeout)
        try:
            return bool(asyncio.run(_go()))
        except Exception as exc:                    # timeout, network, cancelled...
            log(f"TTS request failed ({type(exc).__name__}: {exc})", "warn")
            return False

    return synth


def _tts_key(text: str, voice: str, rate: str, pitch: str, trim: bool) -> str:
    return sha_short(ENGINE, text, voice, rate, pitch, bool(trim))


def _meta_path(wav: str) -> str:
    return os.path.splitext(wav)[0] + ".json"


def cached_clip_duration(wav: str, key: str) -> Optional[float]:
    """Duration of a valid cached clip, or None if it must be regenerated."""
    ok, meta = read_json(_meta_path(wav), ("key", "duration", "bytes"))
    if not ok or meta["key"] != key:
        return None
    try:
        if not os.path.isfile(wav) or os.path.getsize(wav) != int(meta["bytes"]):
            return None
        dur = audio_duration(wav)                   # opens the WAV: proves it is readable
        if dur < MIN_CLIP_SECONDS or abs(dur - float(meta["duration"])) > 0.01:
            return None
        return dur
    except (OSError, TypeError, ValueError):
        return None


def _convert_plain(src: str, dst: str) -> bool:
    try:
        run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "24000",
             "-c:a", "pcm_s16le", dst], quiet=True)
        return os.path.isfile(dst) and os.path.getsize(dst) > 44
    except Exception:
        return False


def _postprocess(raw: str, final_wav: str, trim: bool) -> float:
    """raw audio -> trimmed mono WAV at ``final_wav``. Returns duration (0 on failure)."""
    from ..video import trim_silence

    tmp = final_wav[:-4] + ".tmp.wav"
    done = False
    if trim:
        raw_len = audio_duration(raw)              # WAV via stdlib, MP3 via ffprobe
        used = trim_silence(raw, tmp)               # silence at both ends only; keeps padding
        # Fall back to a plain conversion if trimming failed or over-trimmed.
        done = (used == tmp and audio_duration(tmp) >= MIN_CLIP_SECONDS
                and (raw_len <= 0.0 or audio_duration(tmp) >= raw_len * 0.3))
    if not done:
        if not _convert_plain(raw, tmp):
            return 0.0
    dur = audio_duration(tmp)
    if dur < MIN_CLIP_SECONDS:
        return 0.0
    os.replace(tmp, final_wav)
    return dur


def _generate(seg: Dict[str, Any], ws: Workspace, tts_cfg: Dict[str, Any],
              synth: SynthFn, key: str) -> TtsResult:
    sid = seg["id"]
    text = seg["translated_text"]
    voice, fallback = tts_cfg["voice"], tts_cfg.get("fallback_voice") or ""
    pitch, rate = tts_cfg["pitch"], tts_cfg["rate"]
    trim = bool(tts_cfg.get("trim_silence", True))
    wav = ws.tts_wav(sid)
    raw = os.path.join(ws.tts_raw_dir, f"segment_{sid:04d}.mp3")
    for path in (wav, _meta_path(wav)):             # never leave stale files behind
        if os.path.exists(path):
            os.remove(path)

    used = voice
    ok = synth(text, voice, pitch, rate, raw) and os.path.isfile(raw) and os.path.getsize(raw) > 0
    if not ok and fallback and fallback != voice:
        log(f"Segment {sid}: main voice failed, trying fallback voice {fallback}.", "warn")
        used = fallback
        ok = synth(text, fallback, pitch, rate, raw) and os.path.isfile(raw) \
            and os.path.getsize(raw) > 0
    if not ok:
        return TtsResult(sid, None, failed=True, reason="synthesis failed after retries")

    dur = _postprocess(raw, wav, trim)
    if dur <= 0.0:
        return TtsResult(sid, None, failed=True, reason="generated audio was empty/unreadable")
    atomic_write_json(_meta_path(wav), {
        "id": sid, "key": key, "duration": round(dur, 4), "bytes": os.path.getsize(wav),
        "voice": voice, "voice_used": used, "engine": ENGINE, "rate": rate, "pitch": pitch,
        "text": text})
    try:
        os.remove(raw)
    except OSError:
        pass
    return TtsResult(sid, wav, duration=dur, voice_used=used)


def run_tts(ws: Workspace, cfg: Dict[str, Any], records: List[Dict[str, Any]],
            synth: Optional[SynthFn] = None, force: bool = False
            ) -> Tuple[List[TtsResult], Dict[str, Any]]:
    """Synthesize (or reuse) one clip per segment. Returns (results in id order, info)."""
    tts_cfg = cfg["tts"]
    if str(tts_cfg.get("engine", ENGINE)).lower() != ENGINE:
        raise PipelineError(
            f"tts.engine '{tts_cfg.get('engine')}' is not supported in the English pipeline; "
            "use 'edge'. (CapCut/VieNeu are Vietnamese-only legacy engines.)")
    os.makedirs(ws.tts_dir, exist_ok=True)
    os.makedirs(ws.tts_raw_dir, exist_ok=True)
    trim = bool(tts_cfg.get("trim_silence", True))
    voice, rate, pitch = tts_cfg["voice"], tts_cfg["rate"], tts_cfg["pitch"]

    results: Dict[int, TtsResult] = {}
    todo: List[Tuple[Dict[str, Any], str]] = []
    for rec in records:
        text = (rec.get("translated_text") or "").strip()
        sid = rec["id"]
        if not any(c.isalnum() for c in text):
            reason = "empty translation" if rec.get("flag") == "empty_translation" \
                else "nothing speakable"
            results[sid] = TtsResult(sid, None, skipped=True, reason=reason)
            continue
        key = _tts_key(text, voice, rate, pitch, trim)
        dur = None if force else cached_clip_duration(ws.tts_wav(sid), key)
        if dur is not None:
            results[sid] = TtsResult(sid, ws.tts_wav(sid), duration=dur, cached=True,
                                     voice_used=voice)
        else:
            todo.append((rec, key))

    hits = sum(1 for r in results.values() if r.cached)
    total_speakable = hits + len(todo)
    info: Dict[str, Any] = {"voice": voice, "segments": len(records),
                            "speakable": total_speakable, "cache_hits": hits,
                            "generated": 0, "failed": [], "skipped": [
                                r.id for r in results.values() if r.skipped]}
    if not todo:
        log(f"TTS cache hit: all {hits} speakable clips are valid (voice {voice}).", "ok")
    else:
        if hits:
            log(f"TTS partial cache: {hits} clips reused, {len(todo)} to generate "
                f"(voice {voice}).", "info")
        else:
            log(f"TTS cache miss: generating {len(todo)} clips with {voice}...", "info")
        synth_fn = synth or make_edge_synth(tts_cfg)
        workers = max(1, int(tts_cfg.get("concurrency", 6)))
        done = 0
        lock = threading.Lock()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_generate, rec, ws, tts_cfg, synth_fn, key): rec
                    for rec, key in todo}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:            # a bug/IO error must not kill other clips
                    res = TtsResult(rec["id"], None, failed=True, reason=f"{type(exc).__name__}: {exc}")
                with lock:
                    results[res.id] = res
                    done += 1
                    if res.failed:
                        log(f"Segment {res.id}: TTS FAILED ({res.reason}).", "warn")
                    if done % 10 == 0 or done == len(todo):
                        log(f"  TTS {done}/{len(todo)} generated "
                            f"({time.time() - t0:.0f}s elapsed)", "info")
        info["generated"] = sum(1 for r, _ in todo if not results[r["id"]].failed)

    info["failed"] = sorted(r.id for r in results.values() if r.failed)
    if total_speakable and len(info["failed"]) == total_speakable:
        raise PipelineError(
            "Text-to-speech failed for EVERY segment. edge-tts needs an internet connection, "
            f"and the voice '{voice}' must exist (run: edge-tts --list-voices). "
            "Nothing was dubbed, so the pipeline stops here.")
    if info["failed"]:
        log(f"{len(info['failed'])}/{total_speakable} segments could not be synthesised "
            f"(ids: {info['failed'][:15]}). They will be silent and flagged needs_review.", "warn")
    ordered = [results[r["id"]] for r in records]
    return ordered, info
