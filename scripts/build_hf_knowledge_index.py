"""검수 완료된 Hugging Face 지식 Q&A만 별도 Chroma DB에 인덱싱한다.

기존 rag_corpus 컬렉션과 디렉터리는 변경하지 않는다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLLECTION = "hf_knowledge_info"
DEFAULT_DB = Path(os.environ.get("RAG_HF_KNOWLEDGE_DIR", Path.home() / ".cache/mbti-rag/hf_knowledge_chroma"))
DEFAULT_INPUT = ROOT / "data/processed/hf_knowledge_approved.jsonl"
META = ("title", "doc_type", "category", "source", "source_url", "collected_at", "license", "review_status")


def read_approved(path):
    records, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("review_status") != "approved":
            raise ValueError(f"Unreviewed record at line {number}: {item.get('id')}")
        if item.get("doc_type") != "info":
            raise ValueError(f"Unexpected doc_type at line {number}")
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(f"Missing record ID at line {number}")
        if item["id"] in seen:
            raise ValueError(f"Duplicate record ID: {item['id']}")
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"Missing text at line {number}")
        if any(not isinstance(item.get(key), str) or not item[key] for key in META):
            raise ValueError(f"Missing metadata at line {number}")
        seen.add(item["id"])
        records.append(item)
    if not records:
        raise ValueError("No approved records found")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true", help="기존 실험 컬렉션만 교체")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    records = read_approved(args.input)
    import chromadb
    from rag.retrieve import embed

    args.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    existing = {collection.name for collection in client.list_collections()}
    if COLLECTION in existing:
        if not args.replace:
            raise SystemExit("Experimental collection already exists. Use --replace to rebuild.")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    try:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=embed([item["text"] for item in batch], batch_size=min(16, args.batch_size)),
                metadatas=[{key: item[key] for key in META} for item in batch],
            )
            print(f"Indexed {collection.count()}/{len(records)}")
    except Exception:
        client.delete_collection(COLLECTION)
        raise
    print(f"Completed: {collection.count()} records. DB: {args.db_dir} / {COLLECTION}")


if __name__ == "__main__":
    main()
