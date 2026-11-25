"""실제 데이터/네트워크 의존성 없이 지식 RAG 전처리와 승인 검사를 테스트한다."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_hf_knowledge_corpus as corpus
import build_hf_knowledge_index as index


class KnowledgePipelineTests(unittest.TestCase):
    def setUp(self):
        self.answer = (
            "Cognitive restructuring examines how thoughts influence emotions "
            "and supports evaluating unhelpful beliefs against balanced evidence."
        )
        self.sample = [
            {"dialogue": [
                {"seeker": "What is cognitive restructuring?", "supporter": self.answer},
                {"seeker": "What is cognitive restructuring?", "supporter": self.answer},
            ]},
            [
                {"seeker": "What are the parts of a neuron?", "supporter": self.answer},
                {"seeker": "How do I manage suicidal thoughts?", "supporter": self.answer},
            ],
        ]

    def test_nested_qa_filter_and_deduplication(self):
        records, stats = corpus.make_records(self.sample)
        self.assertEqual(stats["qa_seen"], 4)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["category"], "hf_knowledge/cbt_psychoeducation")
        self.assertEqual(stats["duplicate"], 1)
        self.assertEqual(stats["off_topic"], 1)
        self.assertEqual(stats["high_risk_excluded"], 1)

    def test_index_requires_review_approval(self):
        records, _ = corpus.make_records(self.sample)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            corpus.write_jsonl(path, records)
            with self.assertRaises(ValueError):
                index.read_approved(path)
            records[0]["review_status"] = "approved"
            corpus.write_jsonl(path, records)
            self.assertEqual(len(index.read_approved(path)), 1)

    def test_index_rejects_duplicate_ids(self):
        records, _ = corpus.make_records(self.sample)
        records[0]["review_status"] = "approved"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            corpus.write_jsonl(path, [records[0], records[0]])
            with self.assertRaises(ValueError):
                index.read_approved(path)


if __name__ == "__main__":
    unittest.main()
