"""Guards: the English pipeline must not print Vietnamese, even from the reused modules."""
import ast
import os
import re
import unittest

from tests.helpers import ROOT
from tests.test_pipeline import PipelineBase, needs_ffmpeg

VI = re.compile("[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")

# Functions executed by a DubFlow run (download, audio, ASR, mix, render, helpers).
ON_PATH = {
    "autodub/video.py": {"extract_audio", "ensure_audio", "assemble_timeline_audio",
                         "_assemble_ffmpeg_chunked", "_mix_batch", "render_final", "_run_render",
                         "change_speed", "trim_silence"},
    "autodub/downloader.py": {"download_video", "_validate_download_file", "_download_speed_options",
                              "_build_download_command", "_parse_progress_line", "_ytdlp_cmd",
                              "_cookie_browser_label"},
    "autodub/asr.py": {"transcribe", "_rescue_gaps", "_asr_faster_whisper", "_dispatch",
                       "ensure_speech_map", "speech_map_from_audio"},
    "autodub/utils.py": {"run", "ffmpeg_dir_to_path", "start_file_log", "log"},
}
MESSAGE_CALLS = {"log", "print", "RuntimeError", "ValueError", "Exception"}


# Vietnamese written WITHOUT diacritics (a diacritic regex alone misses these).
VI_PLAIN = re.compile(r"\b(xong trong|khong|duoc|lenh|tiet kiem|tieng|nen bi|bi dung|chay qua|"
                      r"on dinh|dang chay|huy lenh|bo qua|thu lai|vung co)\b", re.I)


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


class StaticScanTest(unittest.TestCase):
    def test_no_vietnamese_messages_in_functions_a_dubflow_run_executes(self):
        offenders = []
        for rel, names in ON_PATH.items():
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src)
            docs = _docstring_nodes(tree)
            for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names):
                for node in ast.walk(fn):
                    if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                            and id(node) not in docs
                            and (VI.search(node.value) or VI_PLAIN.search(node.value))):
                        offenders.append(f"{rel}:{node.lineno} in {fn.name}(): {node.value[:50]!r}")
        self.assertEqual(sorted(set(offenders)), [], "Vietnamese text on the DubFlow path")

    def test_scanner_itself_detects_both_accented_and_plain_vietnamese(self):
        self.assertTrue(VI.search("Ghép audio"))
        self.assertTrue(VI_PLAIN.search("Render xong trong 3s"))
        self.assertTrue(VI_PLAIN.search("Lenh chay qua 60s nen bi dung"))
        self.assertFalse(VI_PLAIN.search("Timeline mix (FFmpeg) finished in 3s"))


@needs_ffmpeg
class RunLogTest(PipelineBase):
    def test_full_run_log_and_console_are_english(self):
        from autodub import utils
        printed = []
        real = utils._safe_print
        utils._safe_print = printed.append
        try:
            r = self.run_pipeline()
        finally:
            utils._safe_print = real
        self.assertTrue(r.ok, r.error)
        with open(os.path.join(self.inter, r.video_id, "pipeline.log"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertGreater(text.count("\n"), 30)
        bad = [ln for ln in text.splitlines() if VI.search(ln)]
        self.assertEqual(bad, [])
        self.assertEqual([ln for ln in printed if VI.search(ln)], [])
        for phrase in ("Mixing 4 clips onto the timeline", "Timeline mix (FFmpeg) finished in",
                       "Final video:", "Batch 1/1: finished in"):
            self.assertIn(phrase, text)
        self.assertEqual([ln for ln in text.splitlines() if VI_PLAIN.search(ln)], [])


if __name__ == "__main__":
    unittest.main()
