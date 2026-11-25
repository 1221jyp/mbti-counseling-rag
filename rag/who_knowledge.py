"""Optional WHO psychoeducation retrieval; no OpenStax ingestion."""
import os
from pathlib import Path

COLLECTION = "who_self_help_2026"
DB_DIR = Path(os.environ.get("RAG_WHO_KNOWLEDGE_DIR", Path.home() / ".cache/mbti-rag/who_knowledge_chroma"))
CORPUS_SHA256 = "c42c2968739ee88e41b2be593b14ad2ac8ce44dcdae70beee0d7c6a2dc9066dd"


def search_who(embedding, n):
    if n <= 0:
        return []
    if not DB_DIR.exists():
        raise FileNotFoundError("Build the local WHO index first.")
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    if (collection.metadata or {}).get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("WHO collection does not match the reviewed corpus. Rebuild it.")
    count = collection.count()
    if not count:
        return []
    result = collection.query(query_embeddings=embedding, n_results=min(n, count))
    return [
        {"id": uid, "text": doc, "score": 1 - dist, **meta}
        for uid, doc, dist, meta in zip(
            result["ids"][0], result["documents"][0],
            result["distances"][0], result["metadatas"][0]
        )
    ]
