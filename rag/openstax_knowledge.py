"""Opt-in OpenStax academic psychoeducation retrieval (local index only)."""
import os
from pathlib import Path

COLLECTION = "openstax_psychology2e_2026"
DB_DIR = Path(os.environ.get("RAG_OPENSTAX_KNOWLEDGE_DIR", Path.home() / ".cache/mbti-rag/openstax_knowledge_chroma"))
CORPUS_SHA256 = "a32d6250716d89263b82872a757b28fa877587a7f8530de1fcee7ce7729ebfeb"


def search_openstax(embedding, n):
    if n <= 0:
        return []
    if not DB_DIR.exists():
        raise FileNotFoundError("Build the local OpenStax index first.")
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    if (collection.metadata or {}).get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("OpenStax collection does not match the reviewed corpus. Rebuild it.")
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
