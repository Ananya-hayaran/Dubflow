"""Stages: mix the dubbed audio track, then render the final MP4.

Mixing places every aligned clip at its ORIGINAL start time on a silent timeline
(never a plain concatenation), so silence between segments is preserved and
overlapping clips are summed with a peak limiter (no hard clipping). Long videos are
mixed in short windows with ffmpeg - nothing is loaded into RAM.

Rendering keeps the original video stream when possible (``-c:v copy``) and encodes
the English audio as AAC. It writes to ``*.partial.mp4`` and only renames it to the
final name after ffprobe validation, so a crash never leaves a fake "finished" file.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils import log
from .alignment import AlignmentItem
from .errors import PipelineError, ValidationError
from .validate import (atomic_write_json, audio_duration, audio_file_ok,
                       measure_max_volume_db, read_json, sha_short, validate_final_mp4)
from .workspace import Workspace

# (clips, starts, total_duration, out_path, sr, mode, chunk_seconds) -> actual output path
MixerFn = Callable[..., str]
# (video, dub_audio, out_path, **render options) -> out_path
RendererFn = Callable[..., str]

SILENT_DB = -70.0          # peak below this means the dub track is effectively silent


def plan_key(items: List[AlignmentItem], sample_rate: int, video_duration: float) -> str:
    """Hash of everything that determines the mixed track."""
    rows = []
    for it in items:
        if it.audio_path and os.path.isfile(it.audio_path):
            rows.append((it.id, round(it.start, 3), round(it.final_duration, 3),
                         round(it.speed_factor, 4), os.path.getsize(it.audio_path)))
    return sha_short(rows, int(sample_rate), round(video_duration, 2))


def run_mix(ws: Workspace, cfg: Dict[str, Any], items: List[AlignmentItem],
            video_duration: float, mixer: Optional[MixerFn] = None,
            force: bool = False) -> Tuple[str, str, Dict[str, Any]]:
    """Returns (dub_audio_path, plan_key, info)."""
    mcfg = cfg["mix"]
    sr = int(mcfg.get("sample_rate", 48000))
    clips = [it for it in items if it.audio_path and os.path.isfile(it.audio_path)]
    if not clips:
        raise PipelineError("There are no dubbed audio clips to mix; refusing to produce a "
                            "video with no English speech.")
    key = plan_key(items, sr, video_duration)

    if not force:
        ok, meta = read_json(ws.dubbed_meta_path, ("key", "path", "duration"))
        if ok and meta["key"] == key and audio_file_ok(
                meta["path"], expect_duration=float(meta["duration"]), tol=0.05):
            log(f"Dubbed audio cache hit: {os.path.basename(meta['path'])} "
                f"({float(meta['duration']):.1f}s, {len(clips)} clips).", "ok")
            return meta["path"], key, {"cache_hit": True, "clips": len(clips),
                                       "duration": float(meta["duration"]),
                                       "max_volume_db": meta.get("max_volume_db")}
        if ok:
            log("Dubbed audio is out of date or unreadable; re-mixing.", "warn")

    if mixer is None:
        from ..video import assemble_timeline_audio as mixer  # type: ignore[assignment]
    log(f"Mixing {len(clips)} clips onto a {video_duration:.1f}s timeline "
        f"(mode={mcfg.get('mode', 'ffmpeg')}, {sr} Hz)...", "info")
    ordered = sorted(clips, key=lambda it: it.start)
    path = mixer([it.audio_path for it in ordered], [it.start for it in ordered],
                 video_duration, ws.dubbed_audio_path, sr=sr,
                 mode=mcfg.get("mode", "ffmpeg"),
                 chunk_seconds=float(mcfg.get("chunk_seconds", 120)))
    dur = audio_duration(path)
    # the mixer pads ~0.2 s of tail; anything far from the video length is a bug
    if dur <= 0 or abs(dur - video_duration) > max(0.5, video_duration * 0.01):
        raise PipelineError(
            f"Mixed audio length {dur:.2f}s does not match the video length "
            f"{video_duration:.2f}s; refusing to continue.")
    peak = measure_max_volume_db(path)
    if peak is not None and peak < SILENT_DB:
        raise PipelineError(f"The mixed dub track is silent (peak {peak:.1f} dB); "
                            "something went wrong while placing the clips.")
    atomic_write_json(ws.dubbed_meta_path, {
        "key": key, "path": path, "duration": round(dur, 4), "video_duration": round(video_duration, 3),
        "sample_rate": sr, "clips": len(clips), "max_volume_db": peak})
    log(f"Dubbed audio written: {path} ({dur:.1f}s, peak {peak if peak is not None else '?'} dB).", "ok")
    return path, key, {"cache_hit": False, "clips": len(clips), "duration": dur,
                       "max_volume_db": peak}


def _final_key(dub_key: str, dub_path: str, video_path: str, rcfg: Dict[str, Any]) -> str:
    return sha_short(dub_key, os.path.getsize(dub_path), os.path.getsize(video_path),
                     rcfg.get("keep_original_db"), bool(rcfg.get("force_h264", True)))


def run_render(ws: Workspace, cfg: Dict[str, Any], video_path: str, dub_path: str,
               dub_key: str, video_duration: float, renderer: Optional[RendererFn] = None,
               force: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Returns (final_mp4_path, info). The returned file has passed validation."""
    rcfg = cfg["render"]
    key = _final_key(dub_key, dub_path, video_path, rcfg)

    if not force:
        ok, meta = read_json(ws.final_meta_path, ("key",))
        if ok and meta["key"] == key and os.path.isfile(ws.final_mp4):
            try:
                report = validate_final_mp4(ws.final_mp4, video_duration)
                log(f"Final video cache hit: {ws.final_mp4} (streams and duration validated).", "ok")
                return ws.final_mp4, {"cache_hit": True, "report": report}
            except ValidationError as exc:
                log(f"Existing final video is not valid ({exc}); re-rendering.", "warn")

    if renderer is None:
        from ..video import render_final as renderer  # type: ignore[assignment]
    os.makedirs(ws.output_dir, exist_ok=True)
    for stale in (ws.partial_mp4,):
        if os.path.exists(stale):
            os.remove(stale)
    renderer(video_path, dub_path, ws.partial_mp4,
             blur_bottom_ratio=0.0, keep_original_db=rcfg.get("keep_original_db"),
             use_gpu=bool(rcfg.get("use_gpu", True)),
             force_h264=bool(rcfg.get("force_h264", True)),
             x264_preset=str(rcfg.get("x264_preset", "veryfast")),
             cpu_threads=int(rcfg.get("cpu_threads", 4)))
    try:
        report = validate_final_mp4(ws.partial_mp4, video_duration)
    except ValidationError:
        try:
            os.remove(ws.partial_mp4)
        except OSError:
            pass
        raise
    os.replace(ws.partial_mp4, ws.final_mp4)          # only validated files get the final name
    atomic_write_json(ws.final_meta_path, {"key": key, "path": ws.final_mp4})
    log(f"Final video written: {ws.final_mp4} "
        f"[{report['video_codec']} {report['width']}x{report['height']} + {report['audio_codec']}, "
        f"{report['duration']:.1f}s]", "ok")
    return ws.final_mp4, {"cache_hit": False, "report": report}
