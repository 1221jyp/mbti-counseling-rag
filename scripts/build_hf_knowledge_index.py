"""검증 근거 목록과 일치하는 심리교육 Q&A만 별도 Chroma DB로 임베딩한다.

검증되지 않은 데이터는 자동으로 차단한다. --replace는 오직 실험 컬렉션에만 적용된다.
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_hf_knowledge_corpus import DEFAULT_MANIFEST, read_manifest, stable_id

COLLECTION = "hf_knowledge_info"
DEFAULT_DB = Path(os.environ.get(
    "RAG_HF_KNOWLEDGE_DIR", Path.home() / ".cache/mbti-rag/hf_knowledge_chroma"
))
DEFAULT_INPUT = ROOT / "data/processed/hf_knowledge_approved.jsonl"
META = (
    "title", "doc_type", "category", "source", "source_url", "collected_at",
    "license", "review_status", "verification_level", "evidence_url",
    "evidence_section", "evidence_note", "evidence_checked_at", "text_sha256",
)


def read_approved(path, manifest_path=DEFAULT_MANIFEST):
    _, references = read_manifest(manifest_path)
    records, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        uid = item.get("id")
        ref = references.get(uid)
        if ref is None or item.get("review_status") != "source_crosschecked":
            raise ValueError(f"Unverified / unlisted entry at line {number}: {uid}")
        if item.get("doc_type") != "info":
            raise ValueError(f"Unexpected doc_type at line {number}")
        if uid in seen:
            raise ValueError(f"Duplicate entry at line {number}: {uid}")
        text = item.get("text")
        if not isinstance(text, str) or not text.startswith("Question: ") or "\nAnswer: " not in text:
            raise ValueError(f"Missing original question/answer structure at line {number}")
        question, answer = text.removeprefix("Question: ").split("\nAnswer: ", 1)
        if uid != stable_id(question, answer):
            raise ValueError(f"Original Q&A has changed at line {number}")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != item.get("text_sha256"):
            raise ValueError(f"Text digest mismatch at line {number}")
        if item.get("evidence_url") != ref["reference_url"] or item.get("evidence_section") != ref["reference_section"]:
            raise ValueError(f"Source citation differs from verified manifest at line {number}")
        if item.get("verification_level") != "concept_source_correlated_not_clinically_validated":
            raise ValueError(f"Unsupported verification level at line {number}")
        if any(not isinstance(item.get(k), str) or not item[k] for k in META):
            raise ValueError(f"Missing provenance metadata at line {number}")
        seen.add(uid)
        records.append(item)
    if not records:
        raise ValueError("No source-crosschecked records found. Nothing to embed.")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true", help="실험용 hf_knowledge_info만 재구축")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    records = read_approved(args.input, args.manifest)
    import chromadb
    from rag.retrieve import embed

    args.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION in existing and not args.replace:
        raise SystemExit("Experimental collection exists. Use --replace to remove old/unverified items.")

    # Build staging first; leave the old collection untouched if embedding fails.
    staging = COLLECTION + "_staging_" + uuid.uuid4().hex[:8]
    collection = client.create_collection(staging, metadata={"hnsw:space": "cosine"})
    try:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=embed([item["text"] for item in batch], batch_size=min(16, args.batch_size)),
                metadatas=[{key: item[key] for key in META} for item in batch],
            )
            print(f"Verified items indexed: {collection.count()}/{len(records)}")
        if collection.count() != len(records):
            raise RuntimeError("Index count mismatch; existing collection left intact")
    except Exception:
        client.delete_collection(staging)
        raise

    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    collection.modify(name=COLLECTION)
    print(f"Completed: {len(records)} source-crosschecked Q&A in {args.db_dir} / {COLLECTION}")
    print("Unverified Q&A were excluded; this does not constitute clinical validation.")


if __name__ == "__main__":
    main()
