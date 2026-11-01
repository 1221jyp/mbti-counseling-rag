"""rag_corpus.jsonl을 임베딩해 Chroma 컬렉션을 새로 만든다."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.retrieve import CHROMA_DIR, COLLECTION, embed, get_client

CORPUS = ROOT / "data/processed/rag_corpus.jsonl"
METADATA_KEYS = ("title", "doc_type", "category", "source", "source_url", "collected_at")
ADD_BATCH = 500


def main():
    records = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    client = get_client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    embeddings = embed([r["text"] for r in records])
    for start in range(0, len(records), ADD_BATCH):
        batch = records[start:start + ADD_BATCH]
        collection.add(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            embeddings=embeddings[start:start + ADD_BATCH],
            metadatas=[{key: r[key] for key in METADATA_KEYS} for r in batch],
        )
    print(f"{collection.count()}개 청크 인덱싱 완료: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
