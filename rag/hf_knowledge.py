"""검수된 Hugging Face 지식 항목에 대한 선택적 검색.

기존 rag_corpus 컬렉션을 변경하지 않는다.
"""
import os
from pathlib import Path

COLLECTION = "hf_knowledge_info"
DB_DIR = Path(os.environ.get("RAG_HF_KNOWLEDGE_DIR", Path.home() / ".cache/mbti-rag/hf_knowledge_chroma"))


def search_knowledge(embedding, n):
    """승인된 전문 지식 항목을 최대 n개 검색한다."""
    if n <= 0:
        return []
    import chromadb

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION not in {collection.name for collection in client.list_collections()}:
        raise FileNotFoundError(
            f"Experimental collection {COLLECTION} does not exist in {DB_DIR}. Build the index first."
        )
    collection = client.get_collection(COLLECTION)
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_embeddings=embedding, n_results=min(n, count))
    return [
        {"id": uid, "text": content, "score": 1 - distance, **metadata}
        for uid, content, distance, metadata in zip(
            result["ids"][0],
            result["documents"][0],
            result["distances"][0],
            result["metadatas"][0],
        )
    ]
