# DubFlow - YouTube URL in, English-dubbed MP4 out

```
python main.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

DubFlow downloads a video, transcribes the speech with timestamps, translates it to
English, speaks it with one consistent English voice, fits every spoken line back into
its original time window, mixes a full-length dub track, and produces:

| Output | Path |
|---|---|
| English-dubbed video | `data/output/<video_id>_dubbed.mp4` |
| English subtitles | `data/output/<video_id>_en.srt` |
| Metrics / stage & cache report | `data/output/<video_id>_metrics.json` |
| Full English dub track | `data/intermediate/<video_id>/dubbed_audio.wav` |

It is built on the AutoDubVN code base (downloader, ASR, timeline mixer, renderer are
reused). The original Vietnamese tool is still in the repo - see
[Legacy Vietnamese tool](#legacy-vietnamese-tool).

## Pipeline

```
[1/10] Downloading video           yt-dlp (prefers H.264 so the video can be copied)
[2/10] Extracting audio            FFmpeg -> 16 kHz mono WAV
[3/10] Transcribing                Faster-Whisper -> transcript.json (ids + timestamps)
[4/10] Translating to English      local NLLB model / passthrough / optional LLM API
[5/10] Generating English TTS      edge-tts, one WAV per segment (segment_0001.wav ...)
[6/10] Aligning audio              speed-fit each clip to its window (FFmpeg atempo)
[7/10] Mixing dubbed audio         clips placed at their ORIGINAL timestamps -> dubbed_audio.wav
[8/10] Rendering final video       original video stream copied + English AAC audio
[9/10] Generating SRT and metrics
[10/10] Validating output          ffprobe checks on the finished MP4
```

## Quick start (Windows)

1. Install **Python 3.10+** and **FFmpeg** (`ffmpeg -version` must work in a new terminal).
2. In the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dubflow.txt
python main.py "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

Local videos work too: `python main.py "C:\videos\talk.mp4"`.

If the video is **not English**, also install the free offline translator once:
`pip install transformers sentencepiece torch`.

## Resuming and caching

Every stage is cached and **validated** before it is trusted (a file merely existing is
not enough): JSON must parse, audio must open with the recorded duration, the final MP4
must contain video and audio streams of the right length. Corrupted files are regenerated.

* Re-run the **same command** after a crash or Ctrl+C - finished stages are reused.
* Only missing/invalid pieces are redone: if 50 of 100 TTS clips are valid, 50 are generated.
* Changing a translation invalidates just that segment's TTS clip, aligned clip, and the mix.
* `--redo tts,align` redoes chosen stages (`download,audio,transcribe,translate,tts,align,mix,render`);
  `--force` redoes everything.

## Timing alignment (how English is fitted into the original timing)

For each segment `target = original_end - original_start` and the TTS clip is measured:

| Situation | Action |
|---|---|
| `tts <= target` (or within 5 %) | natural speed |
| `tts > target` | `speed = tts / target`, applied with FFmpeg `atempo` (pitch preserved) |
| still too long at the cap | **kept whole at 1.25x**, never cut, marked `needs_review`, logged |

Limits: `MIN_TTS_SPEED = 0.85`, `MAX_TTS_SPEED = 1.25`, `ALIGNMENT_TOLERANCE = 0.05`
(see `autodub/dubflow/alignment.py`). Original transcript timestamps are **never**
changed; every clip starts at its original timestamp, silence between lines is preserved,
and overlapping clips are summed through a peak limiter so they cannot hard-clip.
`alignment.json` lists every segment's speed, final length and status
(`ok | adjusted | needs_review | skipped`).

## Translation

| `translation.provider` | Notes |
|---|---|
| `local` (default) | Free, offline, no API key. NLLB-200 (600M) via `transformers`. |
| `passthrough` | No translation. **Used automatically when the source is already English.** |
| `gemini`, `nvidia`, `tokenrouter`, `inferx`, `zenmux` | Optional hosted LLMs; need your own key. Prompt asks for natural, concise, speakable English. |

Segment id, start, end and source text are preserved exactly; only `translated_text` is added.

## Configuration

Nothing is required. To change settings copy `dubflow.example.yaml` to `dubflow.yaml`
(voice, Whisper model size, download quality, speed limits, ...). Handy flags:
`--voice en-US-GuyNeural`, `--model medium`, `--language es`, `--quality 480`,
`--provider passthrough`, `--data-dir D:\dubs`.

Default voice: **`en-US-AriaNeural`** (fallback `en-US-GuyNeural`). No voice cloning.

## Long videos

Work is segment- and chunk-based on disk: TTS and alignment are per-clip files, and the
timeline is mixed with FFmpeg in 120-second windows, so nothing large is held in RAM.
Designed for short clips up to roughly an hour; it does not promise unlimited length.

## Tests

```bat
python -m unittest discover -s tests -v
```

`python tests/offline_demo.py` runs the whole pipeline twice on a synthetic video and prints
the stage/cache table. See `docs/DUBFLOW_AUDIT.md` for what has and has not been verified.

The suite needs only FFmpeg - no network, no model downloads, no paid APIs. It uses real
FFmpeg on synthetic media with fake ASR/translation/TTS engines to test alignment,
placement, silence, overlap, caching, corruption recovery, resume and final-MP4 validation.

## Limitations

* TTS uses `edge-tts` (free, but needs an internet connection).
* The default local translator downloads a ~2.4 GB model the first time a non-English
  video is dubbed; its output length is not controlled, so long lines rely on alignment.
* One voice for all speakers (no diarization, no voice cloning, no lip sync).
* Lines that cannot fit at 1.25x are kept intact and flagged `needs_review`; they may
  overlap the next line rather than being cut.

## Legacy Vietnamese tool

The original AutoDubVN CLI is `main_vi_legacy.py` and its GUI is `gui/app.py`; they read
`config.yaml` (from `config.example.yaml`) and are documented in `README_VI_LEGACY.md`.
They were not converted. Their shared library modules (`video.py`, `downloader.py`,
`asr.py`, `utils.py`) now log in English.
