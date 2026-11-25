"""Index only the fixed, source-selected OpenStax 2026 corpus.

Local non-commercial academic RAG only; the user reported receiving permission.
Run only if the actual written permission covers this use.
"""
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_SHA256 = "a32d6250716d89263b82872a757b28fa877587a7f8530de1fcee7ce7729ebfeb"
PDF_SHA256 = "2f08113c78127bc684922ad6a819db22bd4705ce9fcf28a07f36a2041b1474e2"
DEFAULT_INPUT = ROOT / "data/processed/openstax_selected.jsonl"
REQUIRED = (
    "id", "text", "title", "doc_type", "category", "source", "source_url",
    "license", "citation", "attribution_notice", "section", "language",
    "review_status", "scope", "collected_at", "pdf_page_start", "pdf_page_end",
    "printed_page_start", "printed_page_end", "pdf_sha256", "text_sha256",
)


def validate(path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Corpus SHA-256 mismatch; use the original supplied file.")
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(records) != 40 or len({r["id"] for r in records}) != 40:
        raise ValueError("Expected 40 distinct selected records")
    for r in records:
        if any(key not in r for key in REQUIRED):
            raise ValueError("Missing metadata")
        if not (r["id"].startswith("openstax-2026-")
                and r["doc_type"] == "info"
                and r["category"].startswith("textbook/")
                and r["source_url"].startswith("https://openstax.org/books/psychology-2e/pages/")
                and r["license"] == "CC BY-NC-SA 4.0"
                and r["pdf_sha256"] == PDF_SHA256
                and r["review_status"] == "source_selected_not_clinically_validated"
                and r["scope"] == "noncommercial_academic_psychoeducation"
                and "Access for free at openstax.org" in r["attribution_notice"]
                and 1 <= r["pdf_page_start"] <= r["pdf_page_end"] <= 755
                and r["printed_page_start"] == r["pdf_page_start"] - 12
                and r["printed_page_end"] == r["pdf_page_end"] - 12
                and r["text_sha256"] == hashlib.sha256(r["text"].encode("utf-8")).hexdigest()):
            raise ValueError("Invalid source or record integrity: " + r["id"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--permission-confirmed", action="store_true",
                        help="Confirm written permission covers local extraction and RAG embedding")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    records = validate(args.input)
    print(f"Validated {len(records)} source-selected records (not clinical validation).")
    if args.validate_only:
        return
    if not args.permission_confirmed:
        parser.error("Confirm the permission covers this exact use with --permission-confirmed.")
    import chromadb
    from rag.retrieve import embed
    from rag.openstax_knowledge import DB_DIR, COLLECTION, CORPUS_SHA256
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    names = {getattr(c, "name", c) for c in client.list_collections()}
    if COLLECTION in names and not args.replace:
        raise SystemExit("Collection exists. Use --replace to rebuild only the OpenStax index.")
    token = uuid.uuid4().hex[:8]
    staging = COLLECTION + "_staging_" + token
    backup = COLLECTION + "_backup_" + token
    col = client.create_collection(staging, metadata={
        "hnsw:space": "cosine", "corpus_sha256": CORPUS_SHA256})
    try:
        for start in range(0, len(records), 16):
            batch = records[start:start+16]
            col.add(ids=[r["id"] for r in batch],
                    documents=[r["text"] for r in batch],
                    embeddings=embed([r["text"] for r in batch], batch_size=16),
                    metadatas=[{key: r[key] for key in REQUIRED if key not in {"id", "text"}} for r in batch])
        if col.count() != len(records):
            raise RuntimeError("Index count mismatch")
    except Exception:
        client.delete_collection(staging)
        raise
    old = client.get_collection(COLLECTION) if COLLECTION in names else None
    if old is not None:
        old.modify(name=backup)
    try:
        col.modify(name=COLLECTION)
    except Exception:
        if old is not None:
            old.modify(name=COLLECTION)
        raise
    if old is not None:
        client.delete_collection(backup)
    print(f"Indexed {len(records)} OpenStax records: {DB_DIR} / {COLLECTION}")


if __name__ == "__main__":
    main()
