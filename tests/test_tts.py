"""3-4. English TTS: per-segment files, ids, duration measurement, cache, corruption, resume."""
import os
import unittest

from tests.helpers import FakeSynth, TempDirCase, make_cfg, make_tone, needs_ffmpeg

from autodub.dubflow import tts_stage
from autodub.dubflow.errors import PipelineError
from autodub.dubflow.validate import audio_duration, read_json
from autodub.dubflow.workspace import Workspace


def records(texts, start_id=1):
    return [{"id": i, "start": float(i), "end": float(i) + 1.0, "source_text": "x",
             "translated_text": t} for i, t in enumerate(texts, start_id)]


@needs_ffmpeg
class TtsStageTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.ws = Workspace(self.tmp, "vid").ensure()
        self.cfg = make_cfg()
        self.recs = records(["Hello everyone.", "Today we learn something new.", "Thanks!"])

    def run_tts(self, synth, recs=None, cfg=None, force=False):
        return tts_stage.run_tts(self.ws, cfg or self.cfg, recs or self.recs, synth, force=force)

    # ---- generation, ids, naming, duration ----
    def test_one_wav_per_segment_named_by_segment_id(self):
        results, info = self.run_tts(FakeSynth())
        for r in results:
            self.assertEqual(os.path.basename(r.path), f"segment_{r.id:04d}.wav")
            self.assertTrue(r.path.startswith(self.ws.tts_dir))
        self.assertEqual([r.id for r in results], [1, 2, 3])
        self.assertEqual(info["generated"], 3)
        self.assertEqual(info["voice"], "en-US-AriaNeural")

    def test_ids_are_preserved_for_non_contiguous_ids(self):
        recs = records(["Hello there.", "General Kenobi."], start_id=7)
        results, _ = self.run_tts(FakeSynth(), recs)
        self.assertEqual([r.id for r in results], [7, 8])
        self.assertTrue(os.path.isfile(self.ws.tts_wav(7)))
        self.assertTrue(os.path.isfile(self.ws.tts_wav(8)))

    def test_duration_is_measured_from_the_real_file_and_stored(self):
        results, _ = self.run_tts(FakeSynth(durations={"Hello everyone.": 1.5}))
        r = results[0]
        self.assertAlmostEqual(r.duration, audio_duration(r.path), places=3)
        ok, meta = read_json(self.ws.tts_wav(1).replace(".wav", ".json"), ("duration", "key"))
        self.assertTrue(ok)
        self.assertAlmostEqual(meta["duration"], r.duration, places=3)
        # 1.5 s of tone + 0.3 s of padding silence, trimmed back to roughly the tone length
        self.assertGreater(r.duration, 1.5)
        self.assertLess(r.duration, 1.8)

    def test_silence_trimming_shortens_padded_clips(self):
        synth = FakeSynth(durations={"Hello everyone.": 1.0})
        trimmed, _ = self.run_tts(synth, self.recs[:1])
        ws2 = Workspace(self.tmp, "vid2").ensure()
        cfg = make_cfg(tts={"trim_silence": False})
        untrimmed, _ = tts_stage.run_tts(ws2, cfg, self.recs[:1], synth)
        self.assertLess(trimmed[0].duration, untrimmed[0].duration - 0.1)

    def test_voice_and_text_are_sent_to_the_engine(self):
        synth = FakeSynth()
        self.run_tts(synth)
        self.assertEqual({c["voice"] for c in synth.calls}, {"en-US-AriaNeural"})
        self.assertEqual([c["text"] for c in sorted(synth.calls, key=lambda c: c["out"])],
                         [r["translated_text"] for r in self.recs])

    # ---- caching ----
    def test_second_run_regenerates_nothing(self):
        self.run_tts(FakeSynth())
        again = FakeSynth()
        results, info = self.run_tts(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(info["cache_hits"], 3)
        self.assertTrue(all(r.cached for r in results))

    def test_changed_text_or_voice_invalidates_only_what_changed(self):
        self.run_tts(FakeSynth())
        recs = [dict(r) for r in self.recs]
        recs[1]["translated_text"] = "A brand new sentence."
        synth = FakeSynth()
        _, info = self.run_tts(synth, recs)
        self.assertEqual([c["text"] for c in synth.calls], ["A brand new sentence."])
        self.assertEqual(info["cache_hits"], 2)
        # different voice -> everything is stale
        synth2 = FakeSynth()
        cfg = make_cfg(tts={"voice": "en-US-GuyNeural"})
        self.run_tts(synth2, recs, cfg)
        self.assertEqual(len(synth2.calls), 3)

    def test_corrupted_wav_is_regenerated(self):
        self.run_tts(FakeSynth())
        with open(self.ws.tts_wav(2), "wb") as fh:
            fh.write(b"this is not audio at all")
        synth = FakeSynth()
        results, info = self.run_tts(synth)
        self.assertEqual([c["text"] for c in synth.calls], [self.recs[1]["translated_text"]])
        self.assertEqual(info["cache_hits"], 2)
        self.assertGreater(audio_duration(results[1].path), 0.1)

    def test_truncated_wav_is_regenerated(self):
        self.run_tts(FakeSynth())
        path = self.ws.tts_wav(1)
        with open(path, "rb") as fh:
            head = fh.read(200)
        with open(path, "wb") as fh:
            fh.write(head)                       # valid header, missing data
        synth = FakeSynth()
        self.run_tts(synth)
        self.assertEqual(len(synth.calls), 1)

    def test_missing_metadata_or_wav_triggers_regeneration(self):
        self.run_tts(FakeSynth())
        os.remove(self.ws.tts_wav(1).replace(".wav", ".json"))
        os.remove(self.ws.tts_wav(3))
        synth = FakeSynth()
        self.run_tts(synth)
        self.assertEqual(len(synth.calls), 2)

    def test_partial_resume_generates_only_missing_clips(self):
        recs = records([f"Sentence number {i} is here." for i in range(10)])
        self.run_tts(FakeSynth(), recs)
        for sid in (3, 4, 9):
            os.remove(self.ws.tts_wav(sid))
        synth = FakeSynth()
        results, info = self.run_tts(synth, recs)
        self.assertEqual(len(synth.calls), 3)
        self.assertEqual(info["cache_hits"], 7)
        self.assertEqual(info["generated"], 3)
        self.assertTrue(all(os.path.isfile(r.path) for r in results))

    def test_force_regenerates_everything(self):
        self.run_tts(FakeSynth())
        synth = FakeSynth()
        self.run_tts(synth, force=True)
        self.assertEqual(len(synth.calls), 3)

    # ---- failures are never silent ----
    def test_failed_segment_is_reported_not_dropped(self):
        synth = FakeSynth(fail_texts=["Thanks!"])
        results, info = self.run_tts(synth)
        self.assertEqual(len(results), 3)                 # still one result per segment
        self.assertTrue(results[2].failed)
        self.assertIsNone(results[2].path)
        self.assertEqual(info["failed"], [3])
        self.assertFalse(os.path.exists(self.ws.tts_wav(3)))

    def test_fallback_voice_rescues_a_failing_main_voice(self):
        synth = FakeSynth(fail_texts=["Thanks!"], fail_main_voice_only=True)
        results, info = self.run_tts(synth)
        self.assertFalse(results[2].failed)
        self.assertEqual(results[2].voice_used, "en-US-GuyNeural")
        self.assertEqual(info["failed"], [])

    def test_all_segments_failing_stops_the_pipeline(self):
        synth = FakeSynth(fail_texts=[r["translated_text"] for r in self.recs])
        with self.assertRaises(PipelineError) as cm:
            self.run_tts(synth)
        self.assertIn("EVERY segment", str(cm.exception))

    def test_empty_translation_is_skipped_and_marked(self):
        recs = records(["Hello.", ""])
        recs[1]["flag"] = "empty_translation"
        results, info = self.run_tts(FakeSynth(), recs)
        self.assertTrue(results[1].skipped)
        self.assertEqual(results[1].reason, "empty translation")
        self.assertEqual(info["skipped"], [2])

    def test_engine_error_in_one_worker_does_not_kill_the_rest(self):
        class Boom(FakeSynth):
            def __call__(self, text, voice, pitch, rate, out_path):
                if text == "Thanks!":
                    raise RuntimeError("driver crashed")
                return super().__call__(text, voice, pitch, rate, out_path)
        results, info = self.run_tts(Boom())
        self.assertEqual(info["failed"], [3])
        self.assertFalse(results[0].failed)

    def test_unsupported_engine_is_rejected_with_explanation(self):
        with self.assertRaises(PipelineError) as cm:
            self.run_tts(FakeSynth(), cfg=make_cfg(tts={"engine": "capcut"}))
        self.assertIn("edge", str(cm.exception))

    def test_edge_synth_without_package_gives_install_hint(self):
        import importlib.util
        if importlib.util.find_spec("edge_tts") is not None:
            self.skipTest("edge-tts is installed here")
        with self.assertRaises(PipelineError) as cm:
            tts_stage.make_edge_synth(self.cfg["tts"])
        self.assertIn("pip install edge-tts", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
