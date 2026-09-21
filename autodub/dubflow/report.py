"""Outputs: English SRT and the metrics/report JSON."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from ..srt_utils import Segment, load_srt_file, save_srt_file
from .validate import atomic_write_json
from .workspace import Workspace


def write_srt(ws: Workspace, records: List[Dict[str, Any]]) -> str:
    """English SRT on the ORIGINAL speech timeline (timestamps are never altered)."""
    segs = [Segment(index=i, start=float(r["start"]), end=float(r["end"]),
                    text=r["translated_text"].strip())
            for i, r in enumerate((r for r in records if r["translated_text"].strip()), 1)]
    os.makedirs(ws.output_dir, exist_ok=True)
    save_srt_file(ws.srt_path, segs)
    back = load_srt_file(ws.srt_path)                  # prove the file we wrote parses
    if len(back) != len(segs):
        raise RuntimeError(f"SRT round-trip mismatch: wrote {len(segs)}, read {len(back)}.")
    return ws.srt_path


def write_metrics(ws: Workspace, metrics: Dict[str, Any]) -> str:
    metrics = dict(metrics)
    metrics["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(ws.metrics_path, metrics)
    return ws.metrics_path
