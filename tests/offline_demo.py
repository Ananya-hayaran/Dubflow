"""Offline end-to-end demo: runs the whole DubFlow pipeline TWICE on a synthetic video.

Real FFmpeg does audio extraction, atempo alignment, timeline mixing, the final render and
ffprobe validation. ASR, translation and TTS are FAKE engines (no network/models), so this
is NOT the real YouTube test - it proves the plumbing, caching and validation.

    python tests/offline_demo.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import FakeASR, FakeSynth, FakeTranslator, make_cfg, make_video  # noqa: E402
from autodub.dubflow.pipeline import Backends, DubFlowPipeline, STAGE_NAMES  # noqa: E402
from autodub.dubflow.validate import probe_media  # noqa: E402

SEGS = [(0.5, 3.0, "hola a todos"),
        (3.5, 5.0, "hoy vamos a aprender algo nuevo y muy largo de decir"),
        (6.0, 9.0, "gracias por ver"),
        (9.2, 13.5, "hasta la proxima")]


def run(cfg, video, backends):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = DubFlowPipeline(cfg, backends=backends).run(video)
    return result, time.time() - t0


def table(title, result, seconds):
    print(f"\n{title}  ->  ok={result.ok}  exit={result.exit_code}  ({seconds:.1f}s)")
    for s in result.stages:
        print(f"  [{s['n']:>2}/10] {s['name']:<28} {s['status']:<14} {s['seconds']:>6.2f}s")


def main():
    tmp = tempfile.mkdtemp(prefix="dubflow_demo_")
    video = make_video(os.path.join(tmp, "clip.mp4"), seconds=14)
    cfg = make_cfg(data_dir=os.path.join(tmp, "data"))
    asr, tl, synth = FakeASR(SEGS, "es"), FakeTranslator(), FakeSynth()
    backends = Backends(transcribe=asr, translator=tl, synth=synth)

    r1, t1 = run(cfg, video, backends)
    table("RUN 1 (cold)", r1, t1)
    calls1 = (asr.calls, len(tl.calls), len(synth.calls))
    r2, t2 = run(cfg, video, backends)
    table("RUN 2 (same command, warm cache)", r2, t2)
    calls2 = (asr.calls, len(tl.calls), len(synth.calls))
    print(f"\nengine calls after run 1 (asr, translate batches, tts clips): {calls1}")
    print(f"engine calls after run 2                                     : {calls2}  "
          f"({'no engine was re-run' if calls1 == calls2 else 'ENGINES RE-RAN'})")

    if r1.ok:
        info = probe_media(r1.final_mp4)
        print(f"\nfinal MP4: {r1.final_mp4}")
        print(f"  video={info.video_codec} {info.width}x{info.height}  audio={info.audio_codec}  "
              f"duration={info.duration:.2f}s  size={info.size} B")
        print(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
                              "-of", "csv=p=0", r1.final_mp4], capture_output=True, text=True).stdout.strip())
        print("  needs_review segments:", r1.metrics.get("needs_review"))
        print(f"  SRT: {r1.srt_path}\n  metrics: {r1.metrics_path}")
    ok = r1.ok and r2.ok and calls1 == calls2 and \
        all(s["status"] == "cache hit" for s in r2.stages if 2 <= s["n"] <= 8)
    print("\nRESULT:", "PASS (synthetic)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
