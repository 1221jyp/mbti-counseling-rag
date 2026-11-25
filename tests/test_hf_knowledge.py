"""검증 근거 없는 Q&A의 임베딩 차단을 데이터/네트워크 없이 확인한다."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_hf_knowledge_corpus as corpus
from scripts import build_hf_knowledge_index as index


class VerifiedKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.question = "What is problem-focused coping?"
        self.answer = (
            "Problem-focused coping means trying to address the source of stress "
            "by identifying a practical problem and planning appropriate solutions."
        )
        self.bad_question = "What are the effects of all stress hormones?"
        self.data = {"dialogues": [
            {"seeker": self.question, "supporter": self.answer},
            {"seeker": self.question, "supporter": self.answer},
            {"seeker": self.bad_question, "supporter": self.answer},
            {"seeker": "How to prevent suicide?", "supporter": self.answer},
        ]}
        self.uid = corpus.stable_id(self.question, self.answer)
        self.evidence = {
            "id": self.uid,
            "reference_url": "https://openstax.org/books/psychology-2e/pages/14-4-regulation-of-stress",
            "reference_section": "Problem-Focused and Emotion-Focused Coping",
            "note": "Verified against published textbook definition",
        }
        self.manifest = {
            "dataset": corpus.DATASET,
            "screening_date": "2026-11-25",
            "records": [self.evidence],
        }

    def test_deduplication_and_high_risk_exclusion(self):
        records, stats = corpus.make_records(self.data)
        self.assertEqual(stats["qa_seen"], 4)
        self.assertEqual(stats["duplicate"], 1)
        self.assertEqual(stats["high_risk"], 1)
        self.assertEqual(len(records), 2)

    def test_only_manifest_matched_entry_passes(self):
        records, _ = corpus.make_records(self.data)
        selected = corpus.select_source_crosschecked(
            records, self.manifest, {self.uid: self.evidence}
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], self.uid)
        self.assertEqual(selected[0]["review_status"], "source_crosschecked")
        self.assertEqual(selected[0]["evidence_url"], self.evidence["reference_url"])

    def test_index_rejects_missing_verification_and_tampering(self):
        records, _ = corpus.make_records(self.data)
        selected = corpus.select_source_crosschecked(
            records, self.manifest, {self.uid: self.evidence}
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
            output = Path(tmp) / "approved.jsonl"

            corpus.write_jsonl(output, records)
            with self.assertRaises(ValueError):
                index.read_approved(output, manifest_path)

            corpus.write_jsonl(output, selected)
            self.assertEqual(len(index.read_approved(output, manifest_path)), 1)

            selected[0]["text"] += " Incorrect extra medical claim."
            corpus.write_jsonl(output, selected)
            with self.assertRaises(ValueError):
                index.read_approved(output, manifest_path)

    def test_manifest_requires_authoritative_https_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            bad = dict(self.manifest)
            bad["records"] = [dict(self.evidence, reference_url="https://example.org/fake")]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                corpus.read_manifest(path)


if __name__ == "__main__":
    unittest.main()
