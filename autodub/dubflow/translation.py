"""Stage: translate the transcript to English -> ``translations.json``.

Contract (checked by tests): every output record keeps the segment ``id``,
original ``start``/``end`` and ``source_text`` UNCHANGED; only
``translated_text`` is added. Timestamps are never touched here.

Providers (all behind the small ``Translator`` interface):
  * PassthroughTranslator - source already English, nothing to translate.
  * LocalTranslator       - free/offline NLLB-200 (or any HF seq2seq model). DEFAULT.
  * LLMTranslator         - optional hosted LLM APIs (Gemini, NVIDIA NIM, ...),
                            reusing the transport code in ``autodub.translate``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..utils import log
from .errors import TranslationError
from .validate import atomic_write_json, read_json
from .workspace import Workspace

TRANSLATIONS_VERSION = 1
TARGET_LANGUAGE = "en"
DEFAULT_LOCAL_MODEL = "facebook/nllb-200-distilled-600M"

# ISO 639-1 (what Whisper reports) -> FLORES-200 codes used by NLLB.
NLLB_CODES = {
    "af": "afr_Latn", "ar": "arb_Arab", "az": "azj_Latn", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn",
    "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "es": "spa_Latn", "et": "est_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr",
    "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl",
    "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mr": "mar_Deva", "ms": "zsm_Latn", "my": "mya_Mymr",
    "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "si": "sin_Sinh", "sk": "slk_Latn",
    "sl": "slv_Latn", "so": "som_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "zh": "zho_Hans",
    "yue": "yue_Hant",
}


def to_nllb_code(lang: str) -> str:
    key = (lang or "").strip().lower().replace("_", "-")
    key = key if key in NLLB_CODES else key.split("-")[0]
    if key not in NLLB_CODES:
        raise TranslationError(
            f"The local translator does not know the language code '{lang}'. "
            f"Set asr.language to one of: {', '.join(sorted(NLLB_CODES))}, or use an LLM provider.")
    return NLLB_CODES[key]


def is_english(lang: str) -> bool:
    return (lang or "").strip().lower().split("-")[0] == "en"


def clean_english(text: str) -> str:
    """Normalise a translated line for speaking: single spaces, no stray markup."""
    t = "".join(ch for ch in str(text or "") if ch not in "\u200b\u200c\u200d\ufeff")
    t = " ".join(t.split())
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'`":
        t = t[1:-1].strip()
    return t


# --------------------------------------------------------------------------- #
#  Translators
# --------------------------------------------------------------------------- #
class Translator:
    name = "base"

    def translate_batch(self, texts: Sequence[str], source_lang: str,
                        budgets: Optional[Sequence[Optional[int]]] = None) -> List[str]:
        raise NotImplementedError


class PassthroughTranslator(Translator):
    name = "passthrough"

    def translate_batch(self, texts, source_lang, budgets=None):
        return [clean_english(t) for t in texts]


class LocalTranslator(Translator):
    """Offline translation with a Hugging Face seq2seq model (default NLLB-200 600M).

    Needs ``pip install transformers sentencepiece torch``. The model (~2.4 GB) is
    downloaded once on first use and cached by Hugging Face.
    """

    def __init__(self, model: str = "auto", device: str = "auto", batch_size: int = 16,
                 max_new_tokens: int = 256):
        self.model_id = DEFAULT_LOCAL_MODEL if str(model or "auto").lower() == "auto" else str(model)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = int(max_new_tokens)
        self._tok = None
        self._model = None
        self._torch = None

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"local:{self.model_id}"

    @property
    def is_nllb(self) -> bool:
        return "nllb" in self.model_id.lower()

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise TranslationError(
                "The local translator needs extra packages:\n"
                "    pip install transformers sentencepiece torch\n"
                "(Only required when the video is NOT in English.) "
                f"Missing: {exc.name}") from exc
        self._torch = torch
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Loading translation model {self.model_id} on {dev} "
            "(first run downloads it; later runs use the local cache)...", "info")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(dev)
        self._model.eval()
        self.device = dev

    def _translate_chunk(self, chunk: List[str], source_lang: str) -> List[str]:
        self._load()
        torch = self._torch
        kwargs: Dict[str, Any] = {}
        if self.is_nllb:
            self._tok.src_lang = to_nllb_code(source_lang)
            kwargs["forced_bos_token_id"] = self._tok.convert_tokens_to_ids("eng_Latn")
        enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=256).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, num_beams=4, max_new_tokens=self.max_new_tokens,
                                       **kwargs)
        return self._tok.batch_decode(out, skip_special_tokens=True)

    def translate_batch(self, texts, source_lang, budgets=None):
        if self.is_nllb:
            to_nllb_code(source_lang)          # fail fast on unsupported languages
        results: List[str] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [str(t) for t in texts[i:i + self.batch_size]]
            got = self._translate_chunk(chunk, source_lang)
            if len(got) != len(chunk):
                raise TranslationError(
                    f"Local model returned {len(got)} lines for {len(chunk)} inputs.")
            results.extend(clean_english(g) for g in got)
        return results


LLM_SYSTEM_PROMPT = (
    "You are a professional dubbing translator. Translate each numbered subtitle line "
    "into natural, conversational English that sounds good when SPOKEN ALOUD by a voice "
    "actor. Preserve the meaning, tone and speaking style of the original speech. "
    "Keep every line CONCISE: it must be speakable within the time given by its "
    "'max_chars' budget, so avoid filler, padding and long formal phrasing; prefer short, "
    "common words. Keep names, numbers and units accurate. Do NOT merge or split lines, "
    "and do NOT add notes, numbering, explanations or quotation marks. "
    "Return ONLY a JSON array of strings with exactly as many items as input lines, "
    "in the same order."
)


class LLMTranslator(Translator):
    """Hosted-LLM translation. Reuses transport from ``autodub.translate``."""

    def __init__(self, provider: str, api_cfg: Dict[str, Any],
                 call: Optional[Callable[..., str]] = None, retries: int = 3):
        self.provider = provider.lower()
        self.api_cfg = dict(api_cfg or {})
        self.retries = max(1, retries)
        self._call = call
        for suffix in ("api_key",):                      # fill missing key from environment
            key = f"{self.provider}_{suffix}"
            if not self.api_cfg.get(key):
                self.api_cfg[key] = os.environ.get(key.upper(), "")
        self.name = f"llm:{self.provider}"  # type: ignore[assignment]

    def _api_call(self, prompt: str) -> str:
        from .. import translate as legacy
        api_key, model, base_url, timeout = legacy.api_params_for_provider(self.api_cfg, self.provider)
        if not api_key:
            raise TranslationError(
                f"Provider '{self.provider}' needs an API key. Put "
                f"'{self.provider}_api_key' under translation.api in dubflow.yaml, or set the "
                f"{self.provider.upper()}_API_KEY environment variable. "
                "(Or use the free default: translation.provider: local)")
        caller = self._call or legacy._api_call
        return caller(prompt, api_key, model, 0.3, provider=self.provider,
                      api_base_url=base_url, api_timeout=timeout)

    def translate_batch(self, texts, source_lang, budgets=None):
        from .. import translate as legacy
        items = []
        for i, t in enumerate(texts):
            row: Dict[str, Any] = {"n": i + 1, "text": str(t)}
            if budgets and budgets[i]:
                row["max_chars"] = int(budgets[i])
            items.append(row)
        prompt = (f"{LLM_SYSTEM_PROMPT}\n\nSource language: {source_lang}. "
                  f"Number of lines: {len(items)}.\n\nINPUT:\n"
                  f"{json.dumps(items, ensure_ascii=False)}\n\nOUTPUT (JSON array only):")
        last = ""
        for attempt in range(1, self.retries + 1):
            raw = self._api_call(prompt)
            parsed = legacy._parse_json_lines(raw or "", len(items))
            if parsed is None:
                # LLMs often add a sentence before/after the array ("Here you go: [...]").
                parsed = legacy._parse_json_lines(_extract_array(raw or ""), len(items))
            if parsed is not None:
                return [clean_english(p) for p in parsed]
            last = (raw or "")[:120]
            log(f"LLM reply was not a JSON array of {len(items)} strings "
                f"(attempt {attempt}/{self.retries}).", "warn")
            time.sleep(min(2.0 * attempt, 6.0))
        raise TranslationError(f"'{self.provider}' did not return usable JSON. Last reply: {last!r}")


def _extract_array(raw: str) -> str:
    """The substring from the first '[' to the last ']' (or the input if there is none)."""
    a, b = raw.find("["), raw.rfind("]")
    return raw[a:b + 1] if 0 <= a < b else raw


def make_translator(cfg: Dict[str, Any], source_lang: str) -> Translator:
    tr = cfg["translation"]
    provider = str(tr.get("provider", "local")).strip().lower()
    if provider == "passthrough":
        return PassthroughTranslator()
    if is_english(source_lang) and not tr.get("force"):
        log("Source language is already English: the translation step is a passthrough "
            "(text is kept as spoken). Set translation.force: true to translate anyway.", "info")
        return PassthroughTranslator()
    if provider == "local":
        return LocalTranslator(tr.get("local_model", "auto"), batch_size=int(tr.get("batch_size", 16)))
    if provider == "browser":
        raise TranslationError(
            "The 'browser' (Gemini web) provider is Vietnamese-only legacy code and is not "
            "available in the English pipeline. Use provider: local (free) or an API provider.")
    if provider in ("gemini", "nvidia", "tokenrouter", "tokenrouter_gemini", "inferx", "zenmux"):
        return LLMTranslator(provider, tr.get("api") or {})
    raise TranslationError(
        f"Unknown translation.provider '{provider}'. "
        "Choose: local | passthrough | gemini | nvidia | tokenrouter | inferx | zenmux")


# --------------------------------------------------------------------------- #
#  Stage
# --------------------------------------------------------------------------- #
def char_budget(start: float, end: float, chars_per_sec: float) -> int:
    return max(12, int(round(max(0.3, end - start) * chars_per_sec * 1.1)))


def reading_pressure(records: List[Dict[str, Any]], chars_per_sec: float) -> Dict[str, Any]:
    """How many lines are likely too long to be spoken in their window at natural speed."""
    over = []
    for r in records:
        dur = max(0.3, float(r["end"]) - float(r["start"]))
        est = len(r["translated_text"]) / max(1.0, chars_per_sec)
        if est > dur * 1.25:
            over.append(r["id"])
    return {"lines": len(records), "likely_too_long": len(over), "ids": over[:20]}


def _record_ok(rec: Any, seg: Dict[str, Any]) -> bool:
    """A cached translation is reusable only if it is for exactly this segment."""
    try:
        return (int(rec["id"]) == seg["id"]
                and abs(float(rec["start"]) - seg["start"]) < 1e-3
                and abs(float(rec["end"]) - seg["end"]) < 1e-3
                and rec["source_text"] == seg["text"]
                and isinstance(rec["translated_text"], str)
                and bool(rec["translated_text"].strip()))
    except (KeyError, TypeError, ValueError):
        return False


def run_translation(ws: Workspace, cfg: Dict[str, Any], transcript: Dict[str, Any],
                    translator: Optional[Translator] = None,
                    force: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Translate (or reuse) every segment. Returns (records, info)."""
    segs = transcript["segments"]
    source_lang = transcript["source_language"]
    tr_cfg = cfg["translation"]

    cached: Dict[int, Dict[str, Any]] = {}
    if not force:
        ok, data = read_json(ws.translations_path, ("segments",))
        if ok and isinstance(data.get("segments"), list):
            by_id = {}
            for rec in data["segments"]:
                if isinstance(rec, dict) and "id" in rec:
                    by_id[rec["id"]] = rec
            for seg in segs:
                rec = by_id.get(seg["id"])
                if rec is not None and _record_ok(rec, seg):
                    cached[seg["id"]] = rec

    pending = [s for s in segs if s["id"] not in cached]
    info: Dict[str, Any] = {"total": len(segs), "cache_hits": len(cached),
                            "translated_now": 0, "provider": None}
    if not pending:
        log(f"Translation cache hit: all {len(segs)} segments already translated.", "ok")
        records = [cached[s["id"]] for s in segs]
        info["provider"] = _stored_provider(ws)
        info["reading_pressure"] = reading_pressure(records, float(tr_cfg["chars_per_sec"]))
        return records, info

    if cached:
        log(f"Translation partial cache: {len(cached)} reused, {len(pending)} to translate.", "info")
    else:
        log(f"Translation cache miss: translating {len(pending)} segments to English...", "info")

    tl = translator or make_translator(cfg, source_lang)
    info["provider"] = tl.name
    cps = float(tr_cfg["chars_per_sec"])
    batch_size = max(1, int(tr_cfg.get("batch_size", 16)))
    results = dict(cached)

    def _save(complete: bool) -> None:
        ordered = [results[s["id"]] for s in segs if s["id"] in results]
        atomic_write_json(ws.translations_path, {
            "version": TRANSLATIONS_VERSION, "video_id": ws.video_id,
            "source_language": source_lang, "target_language": TARGET_LANGUAGE,
            "provider": tl.name, "complete": complete, "segments": ordered})

    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        texts = [s["text"] for s in chunk]
        budgets = [char_budget(s["start"], s["end"], cps) for s in chunk]
        out = tl.translate_batch(texts, source_lang, budgets)
        if len(out) != len(chunk):
            raise TranslationError(
                f"Translator returned {len(out)} lines for {len(chunk)} inputs.")
        for seg, text in zip(chunk, out):
            text = clean_english(text)
            rec = {"id": seg["id"], "start": seg["start"], "end": seg["end"],
                   "source_text": seg["text"], "translated_text": text}
            if not text:
                rec["flag"] = "empty_translation"
                log(f"Segment {seg['id']}: translation came back empty "
                    "(it will be silent and marked needs_review).", "warn")
            results[seg["id"]] = rec
        _save(complete=False)                     # progress survives a crash here
        log(f"  translated {min(i + batch_size, len(pending))}/{len(pending)} segments", "info")

    _save(complete=True)
    info["translated_now"] = len(pending)
    records = [results[s["id"]] for s in segs]
    info["reading_pressure"] = reading_pressure(records, cps)
    rp = info["reading_pressure"]
    if rp["likely_too_long"]:
        log(f"{rp['likely_too_long']}/{rp['lines']} lines are probably too long for their "
            "window at natural speed; alignment will speed them up (max 1.25x) or flag them.", "warn")
    return records, info


def _stored_provider(ws: Workspace) -> Optional[str]:
    ok, data = read_json(ws.translations_path, ("segments",))
    return data.get("provider") if ok else None
