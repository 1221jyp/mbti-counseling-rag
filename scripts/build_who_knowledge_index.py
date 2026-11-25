"""Index the fixed source-reviewed WHO Korean educational adaptation locally."""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep validation independent of ML packages.
EXPECTED_SHA256 = "c42c2968739ee88e41b2be593b14ad2ac8ce44dcdae70beee0d7c6a2dc9066dd"
DEFAULT_INPUT = ROOT / "data/processed/who_self_help_selected.jsonl"
REQUIRED = (
    "id", "text", "title", "doc_type", "category", "source", "source_url",
    "license", "citation", "attribution_notice", "section", "language",
    "review_status", "scope", "collected_at", "pdf_page_start", "pdf_page_end",
    "printed_page_start", "printed_page_end",
)


def validate(path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Corpus SHA-256 mismatch: use the original reviewed JSONL file without editing.")
    items = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(items) != 20 or len({item["id"] for item in items}) != len(items):
        raise ValueError("Unexpected corpus size or duplicate ID")
    for item in items:
        if not all(key in item for key in REQUIRED):
            raise ValueError("Missing metadata")
        if (not item["id"].startswith("who-2026-")
                or not item["category"].startswith("textbook/")
                or item["source_url"] != "https://www.who.int/publications/i/item/9789240120785"
                or item["license"] != "CC BY-NC-SA 3.0 IGO"
                or item["review_status"] != "source_reviewed_not_clinically_validated"
                or item["scope"] != "noncommercial_academic_psychoeducation"
                or not 1 <= item["pdf_page_start"] <= item["pdf_page_end"] <= 156
                or not item["text"].strip()):
            raise ValueError("Unexpected record or invalid source metadata")
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    records = validate(args.input)
    print(f"Validated {len(records)} source-reviewed WHO records (not clinically validated).")
    if args.validate_only:
        return
    import chromadb
    from rag.retrieve import embed
    from rag.who_knowledge import DB_DIR, COLLECTION, CORPUS_SHA256
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    names = {getattr(c, "name", c) for c in client.list_collections()}
    if COLLECTION in names and not args.replace:
        raise SystemExit("Collection already exists. Use --replace to rebuild.")
    token = uuid.uuid4().hex[:8]
    staging = COLLECTION + "_staging_" + token
    backup = COLLECTION + "_backup_" + token
    collection = client.create_collection(staging, metadata={
        "hnsw:space": "cosine", "corpus_sha256": CORPUS_SHA256})
    try:
        for start in range(0, len(records), 16):
            batch = records[start:start+16]
            collection.add(ids=[r["id"] for r in batch],
                documents=[r["text"] for r in batch],
                embeddings=embed([r["text"] for r in batch], batch_size=16),
                metadatas=[{key: r[key] for key in REQUIRED if key not in {"id", "text"}} for r in batch])
        if collection.count() != len(records):
            raise RuntimeError("Index count mismatch")
    except Exception:
        client.delete_collection(staging)
        raise
    old = client.get_collection(COLLECTION) if COLLECTION in names else None
    if old is not None:
        old.modify(name=backup)
    try:
        collection.modify(name=COLLECTION)
    except Exception:
        if old is not None:
            old.modify(name=COLLECTION)
        raise
    if old is not None:
        client.delete_collection(backup)
    print(f"Indexed {len(records)} WHO records into {COLLECTION} at {DB_DIR}")


if __name__ == "__main__":
    main()
