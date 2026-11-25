"""bge-m3 임베딩과 Chroma로 상위 k개 청크를 검색한다."""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "rag_corpus"
INFO_RATIO = 0.4
# OneDrive 동기화 폴더에 SQLite를 두면 잠금 문제가 생길 수 있어 기본 위치를 밖으로 뺀다.
CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", Path.home() / ".cache/mbti-rag/chroma"))


@lru_cache(maxsize=1)
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)


@lru_cache(maxsize=1)
def get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embed(texts, batch_size=32):
    return get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True).tolist()


def default_n_info(k):
    """k=3이면 info 1개, k=5이면 info 2개. k=1은 case만 반환한다."""
    return max(1, round(k * INFO_RATIO)) if k > 1 else 0


def search(collection, embedding, doc_type, n):
    if n == 0:
        return []
    result = collection.query(query_embeddings=embedding, n_results=n, where={"doc_type": doc_type})
    return [
        {"id": id_, "text": text, "score": 1 - distance, **metadata}
        for id_, text, distance, metadata in zip(
            result["ids"][0], result["documents"][0], result["distances"][0], result["metadatas"][0]
        )
    ]


def retrieve(query, k=5, n_info=None, n_expert=0):
    """기존 case/info 검색을 유지하며 opt-in으로 검수된 전문 지식을 추가한다.

    n_expert=0은 기존 동작과 동일하다. n_expert>0은 k개의 결과 중
    해당 개수만큼 별도 지식 컬렉션에 배정한다.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(n_expert, int) or not (0 <= n_expert <= k):
        raise ValueError("n_expert must be between 0 and k")
    baseline_k = k - n_expert
    n_info = default_n_info(baseline_k) if n_info is None else n_info
    if not isinstance(n_info, int) or not (0 <= n_info <= baseline_k):
        raise ValueError("n_info must be between 0 and k - n_expert")
    embedding = embed([query])
    results = []
    if baseline_k:
        collection = get_client().get_collection(COLLECTION)
        results += search(collection, embedding, "case", baseline_k - n_info)
        results += search(collection, embedding, "info", n_info)
    if n_expert:
        from .hf_knowledge import search_knowledge
        results += search_knowledge(embedding, n_expert)
    return sorted(results, key=lambda result: result["score"], reverse=True)
