# DubFlow conversion audit

How AutoDubVN was converted into the English DubFlow pipeline, what was verified, and
what was NOT. Status words are used strictly: **COMPLETE** = verified by tests that ran;
**PARTIAL** = implemented, but a real external service/model was never exercised;
**MISSING** = not done.

## Verification environment (important)

Everything below was developed in a sandbox with **no network access** and without
`yt-dlp`, `edge-tts`, `faster-whisper` or `torch` installed. FFmpeg/ffprobe were real.
Therefore the **real YouTube end-to-end test has NOT been run.** The test-suite uses real
FFmpeg on synthetic media with fake ASR / translation / TTS engines
(`python tests/offline_demo.py` reproduces a two-run demo).

## Architecture: reused / modified / added

| | |
|---|---|
| **Reused as-is** | `downloader.download_video`, `video.ensure_audio`, `asr.transcribe` (faster-whisper path), `video.change_speed` (atempo), `video.assemble_timeline_audio` (timeline mixer, chunked), `video.render_final`, `srt_utils`, `utils.log` |
| **Modified (small, backward compatible)** | `timeline.fit_segments_strict` (+`tolerance`, +`needs_review`; defaults reproduce the original output, verified against git HEAD) - `video._mix_batch` (+peak limiter, 2 ms delay) - `downloader.download_video` (+`format_selector`) - runtime log/error messages on the DubFlow path translated to English in `video.py`, `downloader.py`, `asr.py`, `utils.py` |
| **Added** | `autodub/dubflow/` (config, workspace, validate, transcript, translation, tts_stage, alignment, assemble, report, pipeline), new `main.py`, `dubflow.example.yaml`, `requirements-dubflow.txt`, `tests/` |
| **Preserved, not converted** | Vietnamese CLI (`main_vi_legacy.py`), GUI (`gui/`, `ui/`, `autodub/server/`), `config.example.yaml`, content/story/slideshow features, CapCut/VieNeu TTS, Paraformer/Gemini-browser code |

Design changes versus the original engine: over-long clips are **kept and flagged**
(`needs_review`) instead of being trimmed; TTS clips are cached per segment id with a
content hash (the original re-synthesised everything each run and deleted its temp dir).

## DubFlow requirement -> implementation -> status

| # | Phase | Implementation | Status | Evidence / gap |
|---|---|---|---|---|
| 1 | Project setup + CLI | `main.py`, `dubflow/config.py`, `workspace.py`, `pipeline.py` | COMPLETE | CLI/argument tests; real CLI run fails cleanly with exit 1 |
| 2 | YouTube download + FFmpeg | `pipeline._default_download` -> `downloader.download_video`; `video.ensure_audio` | PARTIAL | Audio extraction + cache verified; yt-dlp never executed (argument plumbing checked against the real signature) |
| 3 | Faster-Whisper transcription | `transcript.py` -> `asr.transcribe` | PARTIAL | id/timestamp/cache logic verified with a fake engine; real Whisper never ran |
| 4 | Translation -> English | `translation.py` (local NLLB / passthrough / LLM) | PARTIAL | Contract, cache, resume, LLM prompt+parsing verified; real NLLB model never loaded |
| 5 | English TTS | `tts_stage.py` -> `tts._synth_one` (edge-tts) | PARTIAL | Files, ids, trim, cache, corruption, fallback voice verified with fake synth; real edge-tts and the voice name `en-US-AriaNeural` never exercised |
| 6 | Timing / alignment | `alignment.py` + `timeline.py` + atempo | COMPLETE | Real FFmpeg; mutation-tested |
| 7 | Audio mixing + final MP4 | `assemble.py` -> `video.assemble_timeline_audio` / `render_final` | COMPLETE | Waveform-level placement/silence/overlap tests; `-c:v copy` verified by packet checksum; synthetic media only |
| 8 | Caching + resume + long videos | validated caches in every stage; chunked mixing | PARTIAL | Caching/resume COMPLETE on synthetic runs; longest test 60 s / 40 segments; nothing near an hour |
| 9 | Metrics + logging + SRT | `report.py`, `pipeline.py`, `pipeline.log` | COMPLETE | Tests |
| 10 | End-to-end testing | `tests/`, `tests/offline_demo.py` | MISSING (real) | Synthetic end-to-end and cache demo pass; the real YouTube run and its second-run cache check have not been executed |

## Vietnamese-specific code

* **Converted:** language/voice/provider defaults (new `dubflow` config), translation prompts
  (new English prompt), output names (`<id>_dubbed.mp4`, `<id>_en.srt`), fallback voice,
  log/error text of the shared modules on the DubFlow path.
* **Left as-is on purpose:** the legacy CLI/GUI and their Vietnamese UI, `tts.py`
  (Vietnamese voices/CapCut/VieNeu, only `_synth_one` is reused), `translate.py` prompts
  (only the transport helpers are reused), `srt_utils` Vietnamese text normalisers.
* Comments and docstrings in shared legacy modules remain Vietnamese.

## Known limitations

* edge-tts needs internet; one voice for every speaker; no diarization/cloning/lip-sync.
* The local translator downloads ~2.4 GB on first use and does not control output length.
* When the source is already English the translation stage is a passthrough
  (this includes the suggested test video "Me at the zoo").
* The mixer converts mono TTS to stereo with a 3 dB level drop (pre-existing behaviour).
* Python 3.10 compatibility was checked mechanically (syntax), not by running on 3.10.
* Not run on Windows.
