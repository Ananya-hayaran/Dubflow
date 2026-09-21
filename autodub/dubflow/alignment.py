"""Stage: timing alignment of English TTS clips to the ORIGINAL speech windows.

For every segment:  target = original_end - original_start.

    tts <= target * (1 + tolerance)  ->  keep natural speed (speed 1.0)
    tts  > target * (1 + tolerance)  ->  speed = tts / target, clamped to
                                         [MIN_TTS_SPEED, MAX_TTS_SPEED] (ffmpeg atempo)

If the clip is STILL longer than its window at the maximum speed it is NOT cut and
NOT pushed later: it keeps the maximum safe speed, is marked ``needs_review`` and
logged. Timestamps are never modified; every clip is placed at its original start.
The planning maths reuses the existing ``timeline.fit_segments_strict``.
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import timeline
from ..utils import log
from .errors import PipelineError
from .tts_stage import TtsResult
from .validate import atomic_write_json, audio_duration, read_json
from .workspace import Workspace

MIN_TTS_SPEED = 0.85
MAX_TTS_SPEED = 1.25
ALIGNMENT_TOLERANCE = 0.05
OVERLAP_EPSILON = 0.05          # ignore sub-50ms overlaps in reports

# (input_wav, output_wav, speed) -> writes output_wav
AtempoFn = Callable[[str, str, float], Any]


@dataclass
class AlignParams:
    min_speed: float = MIN_TTS_SPEED
    max_speed: float = MAX_TTS_SPEED
    tolerance: float = ALIGNMENT_TOLERANCE
    max_overhang: float = 0.0
    stretch_short: bool = False

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "AlignParams":
        a = cfg.get("alignment", {})
        p = cls(
            min_speed=float(a.get("min_speed") or MIN_TTS_SPEED),
            max_speed=float(a.get("max_speed") or MAX_TTS_SPEED),
            tolerance=float(ALIGNMENT_TOLERANCE if a.get("tolerance") is None else a["tolerance"]),
            max_overhang=float(a.get("max_overhang_seconds") or 0.0),
            stretch_short=bool(a.get("stretch_short_segments", False)),
        )
        if not (0.5 <= p.min_speed <= 1.0 <= p.max_speed <= 2.0):
            raise PipelineError(
                f"alignment speeds must satisfy 0.5 <= min_speed <= 1.0 <= max_speed <= 2.0 "
                f"(got min={p.min_speed}, max={p.max_speed}).")
        return p


@dataclass
class AlignmentItem:
    id: int
    start: float
    end: float
    target_duration: float
    tts_duration: float = 0.0
    speed_factor: float = 1.0
    final_duration: float = 0.0
    status: str = "ok"                  # ok | adjusted | needs_review | skipped
    reasons: List[str] = field(default_factory=list)
    overlap_next_s: float = 0.0
    audio_path: Optional[str] = None
    cached: bool = False

    @property
    def placed_start(self) -> float:    # never moved: original timestamp
        return self.start

    def as_dict(self) -> Dict[str, Any]:
        r3 = lambda v: round(float(v), 3)
        return {"id": self.id, "start": r3(self.start), "end": r3(self.end),
                "target_duration": r3(self.target_duration),
                "tts_duration": r3(self.tts_duration),
                "speed_factor": round(self.speed_factor, 4),
                "final_duration": r3(self.final_duration),
                "status": self.status, "needs_review": self.status == "needs_review",
                "reasons": list(self.reasons), "overlap_next_s": r3(self.overlap_next_s)}


# --------------------------------------------------------------------------- #
#  Pure planning (no I/O) - easy to unit test
# --------------------------------------------------------------------------- #
def plan_alignment(records: List[Dict[str, Any]], tts_durations: Dict[int, float],
                   total_duration: float, params: AlignParams) -> List[AlignmentItem]:
    starts = [float(r["start"]) for r in records]
    ends = [float(r["end"]) for r in records]
    nat = [max(0.0, float(tts_durations.get(r["id"], 0.0))) for r in records]
    placements = timeline.fit_segments_strict(
        starts, nat, max_speed=params.max_speed, min_gap=0.0,
        total_duration=total_duration if total_duration > 0 else None,
        trim_overflow=False, ends=ends, max_overhang=params.max_overhang,
        tolerance=params.tolerance)
    items: List[AlignmentItem] = []
    for i, (rec, pl) in enumerate(zip(records, placements)):
        target = max(0.01, ends[i] - starts[i])
        speed = max(params.min_speed, min(params.max_speed, float(pl.speed)))
        if (params.stretch_short and nat[i] > 0 and speed <= 1.0 + 1e-9
                and nat[i] < target * (1.0 - params.tolerance)):
            speed = max(params.min_speed, nat[i] / target)      # slow down, never below MIN
        item = AlignmentItem(
            id=rec["id"], start=starts[i], end=ends[i], target_duration=target,
            tts_duration=nat[i], speed_factor=speed,
            final_duration=nat[i] / speed if speed > 0 else nat[i])
        items.append(item)
    return items


def window_end(i: int, items: List[AlignmentItem], total_duration: float,
               params: AlignParams) -> float:
    """End of the window segment i may occupy (own end, plus optional borrowed silence)."""
    it = items[i]
    cap = it.end + params.max_overhang
    if i + 1 < len(items):
        return max(it.end, min(items[i + 1].start, cap))
    if total_duration > 0:
        return max(it.end, min(total_duration, cap))
    return it.end


def classify(items: List[AlignmentItem], total_duration: float, params: AlignParams) -> None:
    """Set status/reasons/overlap from MEASURED final durations (in place)."""
    for i, it in enumerate(items):
        it.reasons = [r for r in it.reasons if r.startswith(("no_audio", "atempo_failed"))]
        if it.audio_path is None:
            it.status = "needs_review" if it.reasons else "skipped"
            continue
        win = max(0.01, window_end(i, items, total_duration, params) - it.start)
        if it.final_duration > win * (1.0 + params.tolerance) + 1e-3:
            it.reasons.append("longer_than_window_at_max_speed")
        if total_duration > 0 and it.start + it.final_duration > total_duration + OVERLAP_EPSILON:
            it.reasons.append("extends_past_video_end")
        if i + 1 < len(items):
            over = it.start + it.final_duration - items[i + 1].start
            it.overlap_next_s = max(0.0, over)
            if it.overlap_next_s > OVERLAP_EPSILON:
                it.reasons.append("overlaps_next_segment")
        blocking = [r for r in it.reasons if r != "overlaps_next_segment"]
        if blocking:
            it.status = "needs_review"
        elif abs(it.speed_factor - 1.0) > 1e-3:
            it.status = "adjusted"
        else:
            it.status = "ok"


# --------------------------------------------------------------------------- #
#  Stage
# --------------------------------------------------------------------------- #
def _default_atempo(src: str, dst: str, speed: float) -> None:
    from ..video import change_speed
    change_speed(src, dst, speed)


def _meta(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _aligned_valid(out: str, src: str, speed: float) -> Optional[float]:
    """Duration of a valid cached aligned clip, else None."""
    ok, meta = read_json(_meta(out), ("speed", "in_bytes", "in_duration", "out_bytes", "out_duration"))
    if not ok:
        return None
    try:
        if abs(float(meta["speed"]) - speed) > 1e-4:
            return None
        if os.path.getsize(src) != int(meta["in_bytes"]):
            return None
        if abs(audio_duration(src) - float(meta["in_duration"])) > 0.01:
            return None
        if not os.path.isfile(out) or os.path.getsize(out) != int(meta["out_bytes"]):
            return None
        dur = audio_duration(out)
        if dur < 0.02 or abs(dur - float(meta["out_duration"])) > 0.01:
            return None
        return dur
    except (OSError, TypeError, ValueError):
        return None


def _process(item: AlignmentItem, tts: TtsResult, ws: Workspace, atempo: AtempoFn,
             force: bool) -> None:
    src = tts.path
    out = ws.aligned_wav(item.id)
    speed = item.speed_factor
    dur = None if force else _aligned_valid(out, src, speed)
    if dur is not None:
        item.cached, item.audio_path, item.final_duration = True, out, dur
        return
    for path in (out, _meta(out)):
        if os.path.exists(path):
            os.remove(path)
    if abs(speed - 1.0) <= 1e-3:
        shutil.copyfile(src, out)
    else:
        try:
            atempo(src, out, speed)                 # ffmpeg atempo (pitch preserved)
            if audio_duration(out) < 0.02:
                raise RuntimeError("atempo produced an empty file")
        except Exception as exc:
            # Keep the segment (un-sped) instead of losing it, and flag it.
            log(f"Segment {item.id}: atempo failed ({exc}); keeping natural speed, "
                "marked needs_review.", "warn")
            shutil.copyfile(src, out)
            speed = item.speed_factor = 1.0
            item.final_duration = audio_duration(src)
            item.reasons.append("atempo_failed")
    dur = audio_duration(out)
    if dur < 0.02:
        raise PipelineError(f"Aligned clip for segment {item.id} is empty/unreadable.")
    atomic_write_json(_meta(out), {
        "id": item.id, "speed": round(speed, 6), "in_bytes": os.path.getsize(src),
        "in_duration": round(audio_duration(src), 4), "out_bytes": os.path.getsize(out),
        "out_duration": round(dur, 4)})
    item.audio_path, item.final_duration = out, dur


def run_alignment(ws: Workspace, cfg: Dict[str, Any], records: List[Dict[str, Any]],
                  tts_results: List[TtsResult], total_duration: float,
                  atempo: Optional[AtempoFn] = None, force: bool = False
                  ) -> Tuple[List[AlignmentItem], Dict[str, Any]]:
    params = AlignParams.from_cfg(cfg)
    os.makedirs(ws.aligned_dir, exist_ok=True)
    by_id = {t.id: t for t in tts_results}
    durations = {t.id: t.duration for t in tts_results if t.path}
    items = plan_alignment(records, durations, total_duration, params)

    todo = []
    for it in items:
        t = by_id.get(it.id)
        if t is None or not t.path:
            if t is not None and t.failed:
                it.reasons.append("no_audio:tts_failed")
            elif t is not None and t.reason == "empty translation":
                it.reasons.append("no_audio:empty_translation")
            continue
        todo.append((it, t))

    atempo_fn = atempo or _default_atempo
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_process, it, t, ws, atempo_fn, force) for it, t in todo]
        for f in futs:
            f.result()                              # surfaces exceptions
    classify(items, total_duration, params)

    counts = {"ok": 0, "adjusted": 0, "needs_review": 0, "skipped": 0}
    for it in items:
        counts[it.status] += 1
    cache_hits = sum(1 for it in items if it.cached)
    audio_items = [it for it in items if it.audio_path]
    info: Dict[str, Any] = {
        "segments": len(items), "with_audio": len(audio_items), "cache_hits": cache_hits,
        "processed": len(audio_items) - cache_hits, **counts,
        # counted by actual speed, independent of status (a needs_review clip that was
        # sped up to the maximum is still "sped up")
        "sped_up": sum(1 for it in audio_items if it.speed_factor > 1.001),
        "max_speed_used": round(max((it.speed_factor for it in audio_items), default=1.0), 3),
        "overlaps": sum(1 for it in items if it.overlap_next_s > OVERLAP_EPSILON),
        "total_overlap_s": round(sum(it.overlap_next_s for it in items), 3),
        "params": {"min_speed": params.min_speed, "max_speed": params.max_speed,
                   "tolerance": params.tolerance, "max_overhang_seconds": params.max_overhang},
    }
    atomic_write_json(ws.alignment_path, {
        "version": 1, "video_id": ws.video_id, "video_duration": round(total_duration, 3),
        "summary": {k: v for k, v in info.items() if k != "params"},
        "params": info["params"], "segments": [it.as_dict() for it in items]})

    if cache_hits == len(audio_items) and audio_items:
        log(f"Alignment cache hit: all {cache_hits} aligned clips are valid.", "ok")
    elif cache_hits:
        log(f"Alignment partial cache: {cache_hits} reused, {info['processed']} processed.", "info")
    log(f"Alignment: {len(audio_items)} clips, {info['sped_up']} sped up "
        f"(max {info['max_speed_used']}x), {counts['needs_review']} needs_review, "
        f"{info['overlaps']} overlapping.", "ok" if not counts["needs_review"] else "warn")
    shown = 0
    for it in items:
        if it.status == "needs_review" and shown < 10:
            shown += 1
            log(f"  needs_review: segment {it.id} @ {it.start:.2f}s window "
                f"{it.target_duration:.2f}s, TTS {it.tts_duration:.2f}s, speed "
                f"{it.speed_factor:.2f}x -> {', '.join(it.reasons)}", "warn")
    if counts["needs_review"] > shown:
        log(f"  ...and {counts['needs_review'] - shown} more (see {os.path.basename(ws.alignment_path)}).",
            "warn")
    return items, info
