#!/usr/bin/env python3
"""DubFlow - turn a YouTube video into an English-dubbed MP4.

    python main.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    python main.py "C:\\videos\\talk.mp4"            (local files work too)

Runs the whole pipeline (download -> speech-to-text -> English translation ->
English TTS -> timing alignment -> mixing -> final MP4 + SRT + metrics).
Run it again with the same URL to resume: finished stages are reused from cache.

The original Vietnamese dubbing CLI is preserved as ``main_vi_legacy.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autodub.dubflow import __version__  # noqa: E402
from autodub.dubflow.config import load_config  # noqa: E402
from autodub.dubflow.errors import PipelineError  # noqa: E402
from autodub.dubflow.pipeline import REDO_NAMES, DubFlowPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", description="Dub a YouTube video (or local video) into English.",
        epilog="Re-running the same command resumes from cached stages.")
    p.add_argument("source", help="YouTube URL (or path to a local video file)")
    p.add_argument("--config", help="path to a config file (default: ./dubflow.yaml if present)")
    p.add_argument("--data-dir", help="folder for downloads/intermediate/output (default: ./data)")
    p.add_argument("--force", action="store_true", help="ignore ALL caches and redo every stage")
    p.add_argument("--redo", default="", metavar="STAGES",
                   help=f"redo only these stages (comma separated): {','.join(REDO_NAMES)}")
    p.add_argument("--voice", help="English voice, e.g. en-US-GuyNeural (default en-US-AriaNeural)")
    p.add_argument("--model", help="Whisper model: tiny|base|small|medium|large-v3 (default small)")
    p.add_argument("--language", help="source language code, e.g. es, hi, ta (default: auto-detect)")
    p.add_argument("--provider", help="translation provider: local|passthrough|gemini|nvidia|...")
    p.add_argument("--quality", help="max download height: best|1080|720|480|360 (default 720)")
    p.add_argument("--version", action="version", version=f"DubFlow {__version__}")
    return p


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.voice:
        cfg["tts"]["voice"] = args.voice
    if args.model:
        cfg["asr"]["model_size"] = args.model
    if args.language:
        cfg["asr"]["language"] = args.language
    if args.provider:
        cfg["translation"]["provider"] = args.provider
    if args.quality:
        cfg["download"]["quality"] = args.quality
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), args)
        redo = tuple(s.strip() for s in args.redo.split(",") if s.strip())
        pipeline = DubFlowPipeline(cfg, force=args.force, redo=redo)
    except (PipelineError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[x] {exc}", file=sys.stderr)
        return 1
    result = pipeline.run(args.source)
    return result.exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled. Re-run the same command to resume.")
        sys.exit(130)
