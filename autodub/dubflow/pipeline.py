"""DubFlow orchestrator: YouTube URL (or local video) -> English-dubbed MP4.

Ten visible stages, each cached and resumable:

  [1/10] Downloading video          [6/10] Aligning audio
  [2/10] Extracting audio           [7/10] Mixing dubbed audio
  [3/10] Transcribing               [8/10] Rendering final video
  [4/10] Translating to English     [9/10] Generating SRT and metrics
  [5/10] Generating English TTS    [10/10] Validating output

Design rules: a stage's cache is trusted only after validation; any failure stops the
run with a clear message and a non-zero exit code; success is reported ONLY after the
final MP4 passes ffprobe validation.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils import (ffmpeg_dir_to_path, log, start_file_log, stop_file_log, which)
from . import __version__
from .alignment import AlignmentItem, AtempoFn, run_alignment
from .assemble import MixerFn, RendererFn, run_mix, run_render
from .config import PROJECT_ROOT
from .errors import PipelineError, ValidationError
from .report import write_metrics, write_srt
from .transcript import Transcriber, run_transcription
from .translation import Translator, run_translation
from .tts_stage import SynthFn, TtsResult, run_tts
from .validate import (audio_duration, audio_file_ok, probe_media, read_json,
                       source_video_ok, validate_final_mp4)
from .workspace import Workspace, derive_video_id, is_url

TOTAL_STAGES = 10
STAGE_NAMES = {
    1: "Downloading video", 2: "Extracting audio", 3: "Transcribing",
    4: "Translating to English", 5: "Generating English TTS", 6: "Aligning audio",
    7: "Mixing dubbed audio", 8: "Rendering final video",
    9: "Generating SRT and metrics", 10: "Validating output",
}
REDO_NAMES = ("download", "audio", "transcribe", "translate", "tts", "align", "mix", "render")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


class StageFailure(Exception):
    def __init__(self, n: int, cause: BaseException):
        super().__init__(str(cause))
        self.n, self.cause = n, cause


@dataclass
class Backends:
    """Optional overrides so tests (and advanced users) can swap a stage's engine."""
    download: Optional[Callable[[str, str, Dict[str, Any]], str]] = None
    transcribe: Optional[Transcriber] = None
    translator: Optional[Translator] = None
    synth: Optional[SynthFn] = None
    atempo: Optional[AtempoFn] = None
    mixer: Optional[MixerFn] = None
    renderer: Optional[RendererFn] = None


@dataclass
class PipelineResult:
    ok: bool
    exit_code: int
    video_id: str
    final_mp4: Optional[str] = None
    srt_path: Optional[str] = None
    metrics_path: Optional[str] = None
    dubbed_audio: Optional[str] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def build_format_selector(dl_cfg: Dict[str, Any]) -> str:
    """yt-dlp -f string. Prefers H.264 (avc1) + AAC so the video stream can be copied."""
    q = str(dl_cfg.get("quality", "720")).strip().lower()
    h = "" if q in ("best", "") else f"[height<={int(q)}]"
    if dl_cfg.get("prefer_h264", True):
        return (f"bv*{h}[vcodec^=avc1]+ba[ext=m4a]/bv*{h}+ba/b{h}[vcodec!=none]")
    return f"bv*{h}+ba/b{h}[vcodec!=none]"


def find_cached_source(source_dir: str) -> Optional[str]:
    """Newest fully-downloaded video in ``source_dir`` that has video AND audio."""
    if not os.path.isdir(source_dir):
        return None
    cands = []
    for name in os.listdir(source_dir):
        low = name.lower()
        if low.endswith(VIDEO_EXTS) and ".part" not in low and ".temp" not in low:
            p = os.path.join(source_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                cands.append(p)
    for p in sorted(cands, key=os.path.getmtime, reverse=True):
        if source_video_ok(p):
            return p
    return None


def _default_download(url: str, dest_dir: str, cfg: Dict[str, Any]) -> str:
    from .. import downloader
    dl = cfg["download"]
    return downloader.download_video(
        url, dest_dir, quality=str(dl.get("quality", "720")),
        cookies_from_browser=dl.get("cookies_from_browser"),
        cookies_file=dl.get("cookies_file"),
        concurrent_fragments=dl.get("concurrent_fragments", 8),
        external_downloader=dl.get("external_downloader", "auto"),
        format_selector=build_format_selector(dl))


def check_tools() -> None:
    """ffmpeg and ffprobe must exist AND run (clear English errors)."""
    for tool in ("ffmpeg", "ffprobe"):
        path = which(tool)
        if not path:
            raise PipelineError(
                f"'{tool}' was not found. Install FFmpeg and add its 'bin' folder to PATH "
                "(Windows: https://www.gyan.dev/ffmpeg/builds/ , or set ffmpeg_dir in dubflow.yaml).")
        try:
            subprocess.run([path, "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20, check=True)
        except Exception as exc:
            raise PipelineError(f"'{tool}' exists at {path} but cannot run: {exc}") from exc


# --------------------------------------------------------------------------- #
#  pipeline
# --------------------------------------------------------------------------- #
class DubFlowPipeline:
    def __init__(self, cfg: Dict[str, Any], backends: Optional[Backends] = None,
                 force: bool = False, redo: Tuple[str, ...] = (), base_dir: Optional[str] = None):
        self.cfg = cfg
        self.backends = backends or Backends()
        self.force = force
        self.redo = set(redo)
        unknown = self.redo - set(REDO_NAMES)
        if unknown:
            raise PipelineError(f"Unknown --redo stage(s): {sorted(unknown)}. "
                                f"Choose from: {', '.join(REDO_NAMES)}")
        data_dir = cfg.get("data_dir", "data")
        base = base_dir or PROJECT_ROOT
        self.data_dir = data_dir if os.path.isabs(data_dir) else os.path.join(base, data_dir)
        self.stages: List[Dict[str, Any]] = []
        self.ctx: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def _forced(self, name: str) -> bool:
        return self.force or name in self.redo

    # ---- stage runner ---- #
    def _stage(self, n: int, fn: Callable[[], Tuple[str, Dict[str, Any]]]) -> None:
        title = STAGE_NAMES[n]
        log(f"[{n}/{TOTAL_STAGES}] {title}...", "step")
        t0 = time.time()
        try:
            status, info = fn()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            self.stages.append({"n": n, "name": title, "status": "failed",
                                "seconds": round(time.time() - t0, 2), "error": str(exc)})
            raise StageFailure(n, exc) from exc
        secs = time.time() - t0
        self.stages.append({"n": n, "name": title, "status": status, "seconds": round(secs, 2)})
        log(f"[{n}/{TOTAL_STAGES}] {title}: {status} ({secs:.1f}s)", "ok")

    # ---- public entry ---- #
    def run(self, source: str) -> PipelineResult:
        source = (source or "").strip().strip('"').strip("'")
        video_id = derive_video_id(source)
        ws = Workspace(self.data_dir, video_id).ensure()
        start_file_log(ws.log_path, append=True)
        t_start = time.time()
        self.metrics = {"tool": "DubFlow", "version": __version__, "video_id": video_id,
                        "source": source, "status": "running",
                        "target_language": "en"}
        try:
            fdir = str(self.cfg.get("ffmpeg_dir") or "").strip()
            if fdir:
                ffmpeg_dir_to_path(fdir)
            check_tools()
            log(f"DubFlow {__version__} | video id: {video_id} | workspace: {ws.root}", "info")
            for n, fn in (
                (1, lambda: self._download(source, ws)),
                (2, lambda: self._audio(ws)),
                (3, lambda: self._transcribe(ws)),
                (4, lambda: self._translate(ws)),
                (5, lambda: self._tts(ws)),
                (6, lambda: self._align(ws)),
                (7, lambda: self._mix(ws)),
                (8, lambda: self._render(ws)),
                (9, lambda: self._reports(ws)),
                (10, lambda: self._validate(ws)),
            ):
                self._stage(n, fn)
        except StageFailure as fail:
            title = STAGE_NAMES[fail.n]
            msg = str(fail.cause) or type(fail.cause).__name__
            log(f"FAILED at stage [{fail.n}/{TOTAL_STAGES}] {title}: {msg}", "err")
            log("No final video was reported. Fix the problem and re-run the SAME command: "
                "completed stages are cached and will be reused.", "err")
            # The Python traceback goes to the log file only; the console keeps the
            # short, human-readable message above.
            import traceback
            try:
                with open(ws.log_path, "a", encoding="utf-8") as fh:
                    fh.write("".join(traceback.format_exception(
                        type(fail.cause), fail.cause, fail.cause.__traceback__)))
            except OSError:
                pass
            log(f"Technical details: {ws.log_path}", "dim")
            self.metrics.update({"status": "failed", "failed_stage": title, "error": msg,
                                 "stages": self.stages,
                                 "total_seconds": round(time.time() - t_start, 2)})
            path = None
            try:
                os.makedirs(ws.output_dir, exist_ok=True)
                path = write_metrics(ws, self.metrics)
            except Exception:
                pass
            stop_file_log()
            return PipelineResult(False, 1, video_id, failed_stage=title, error=msg,
                                  stages=self.stages, metrics=self.metrics, metrics_path=path)
        except KeyboardInterrupt:
            log("Interrupted. Progress so far is cached; re-run the same command to resume.", "warn")
            stop_file_log()
            raise
        except PipelineError as exc:            # setup problems (ffmpeg missing, ...)
            log(f"FAILED before starting: {exc}", "err")
            stop_file_log()
            return PipelineResult(False, 1, video_id, failed_stage="setup", error=str(exc))

        total = time.time() - t_start
        self.metrics.update({"status": "success", "stages": self.stages,
                             "total_seconds": round(total, 2)})
        write_metrics(ws, self.metrics)
        needs = self.metrics.get("alignment", {}).get("needs_review", 0)
        log(f"DONE in {total:.1f}s. Final video validated.", "ok")
        print(f"\n  Dubbed video : {ws.final_mp4}\n  English SRT  : {ws.srt_path}"
              f"\n  Metrics      : {ws.metrics_path}\n  Dub audio    : {self.ctx['dub_path']}")
        if needs:
            print(f"  NOTE: {needs} segment(s) flagged needs_review "
                  f"(see {os.path.basename(ws.alignment_path)}).")
        print()
        stop_file_log()
        return PipelineResult(True, 0, video_id, final_mp4=ws.final_mp4, srt_path=ws.srt_path,
                              metrics_path=ws.metrics_path, dubbed_audio=self.ctx["dub_path"],
                              stages=self.stages, metrics=self.metrics)

    # ---- stages ---- #
    def _download(self, source: str, ws: Workspace):
        if is_url(source):
            cached = None if self._forced("download") else find_cached_source(ws.source_dir)
            if cached:
                log(f"Download cache hit: {cached}", "ok")
                path, status = cached, "cache hit"
            else:
                log("Download cache miss: downloading with yt-dlp...", "info")
                path = (self.backends.download or _default_download)(source, ws.source_dir, self.cfg)
                status = "done"
        else:
            path, status = os.path.abspath(source), "local file"
            if not os.path.isfile(path):
                raise PipelineError(f"File not found: {path}")
        if not source_video_ok(path):
            raise PipelineError(f"'{path}' is not a usable video (needs a video AND an audio "
                                "stream and a non-zero duration).")
        info = probe_media(path)
        self.ctx.update(video_path=path, video_duration=info.duration)
        self.metrics["source_video"] = {"path": path, **info.as_dict()}
        log(f"Video: {os.path.basename(path)} | {info.duration:.1f}s | "
            f"{info.video_codec} {info.width}x{info.height} + {info.audio_codec}", "info")
        return status, {}

    def _audio(self, ws: Workspace):
        from ..video import ensure_audio
        force = self._forced("audio")
        hit = (not force and os.path.isfile(ws.audio_path)
               and audio_file_ok(ws.audio_path, expect_duration=self.ctx["video_duration"], tol=0.5))
        path = ensure_audio(self.ctx["video_path"], ws.audio_path, sr=16000,
                            loudnorm=bool(self.cfg["asr"].get("loudnorm", True)),
                            reuse_existing=not force)
        if not audio_file_ok(path):
            raise PipelineError(f"Audio extraction produced no usable audio: {path}")
        self.ctx.update(audio_path=path, audio_duration=audio_duration(path))
        log(f"Audio: {os.path.basename(path)} ({self.ctx['audio_duration']:.1f}s)", "info")
        return ("cache hit" if hit else "done"), {}

    def _transcribe(self, ws: Workspace):
        data, hit = run_transcription(ws, self.cfg, self.ctx["audio_path"],
                                      self.ctx["audio_duration"], self.backends.transcribe,
                                      force=self._forced("transcribe"))
        self.ctx["transcript"] = data
        self.metrics["transcript"] = {"segments": len(data["segments"]),
                                      "source_language": data["source_language"],
                                      "asr": data.get("asr", {})}
        return ("cache hit" if hit else "done"), {}

    def _translate(self, ws: Workspace):
        records, info = run_translation(ws, self.cfg, self.ctx["transcript"],
                                        self.backends.translator, force=self._forced("translate"))
        self.ctx["records"] = records
        self.metrics["translation"] = info
        if info["translated_now"] == 0:
            return "cache hit", info
        return ("partial cache" if info["cache_hits"] else "done"), info

    def _tts(self, ws: Workspace):
        results, info = run_tts(ws, self.cfg, self.ctx["records"], self.backends.synth,
                                force=self._forced("tts"))
        self.ctx["tts"] = results
        self.metrics["tts"] = info
        if info["generated"] == 0 and not info["failed"]:
            return "cache hit", info
        return ("partial cache" if info["cache_hits"] else "done"), info

    def _align(self, ws: Workspace):
        items, info = run_alignment(ws, self.cfg, self.ctx["records"], self.ctx["tts"],
                                    self.ctx["video_duration"], self.backends.atempo,
                                    force=self._forced("align"))
        self.ctx["items"] = items
        self.metrics["alignment"] = info
        if info["with_audio"] and info["cache_hits"] == info["with_audio"]:
            return "cache hit", info
        return ("partial cache" if info["cache_hits"] else "done"), info

    def _mix(self, ws: Workspace):
        path, key, info = run_mix(ws, self.cfg, self.ctx["items"], self.ctx["video_duration"],
                                  self.backends.mixer, force=self._forced("mix"))
        self.ctx.update(dub_path=path, dub_key=key)
        self.metrics["mix"] = info
        return ("cache hit" if info["cache_hit"] else "done"), info

    def _render(self, ws: Workspace):
        path, info = run_render(ws, self.cfg, self.ctx["video_path"], self.ctx["dub_path"],
                                self.ctx["dub_key"], self.ctx["video_duration"],
                                self.backends.renderer, force=self._forced("render"))
        self.ctx["final"] = path
        self.metrics["final_render"] = {"cache_hit": info["cache_hit"]}
        return ("cache hit" if info["cache_hit"] else "done"), info

    def _reports(self, ws: Workspace):
        srt = write_srt(ws, self.ctx["records"])
        items: List[AlignmentItem] = self.ctx["items"]
        self.metrics["needs_review"] = [it.id for it in items if it.status == "needs_review"]
        self.metrics["voice"] = self.cfg["tts"]["voice"]
        self.metrics["cache_summary"] = [
            {"stage": s["name"], "status": s["status"]} for s in self.stages]
        write_metrics(ws, self.metrics)                  # provisional; finalised after validation
        log(f"English SRT: {srt}", "ok")
        return "done", {}

    def _validate(self, ws: Workspace):
        report = validate_final_mp4(ws.final_mp4, self.ctx["video_duration"])
        if not os.path.isfile(self.ctx["dub_path"]) or audio_duration(self.ctx["dub_path"]) <= 0:
            raise ValidationError("Dubbed audio file is missing or unreadable.")
        if not os.path.isfile(ws.srt_path) or os.path.getsize(ws.srt_path) == 0:
            raise ValidationError("English SRT file is missing or empty.")
        self.metrics["final_video"] = report
        self.metrics["dubbed_audio"] = {"path": self.ctx["dub_path"],
                                        "duration": round(audio_duration(self.ctx["dub_path"]), 3)}
        log(f"Validated: video={report['video_codec']} {report['width']}x{report['height']}, "
            f"audio={report['audio_codec']}, duration={report['duration']:.2f}s "
            f"(source {report['source_duration']:.2f}s).", "ok")
        return "done", report
