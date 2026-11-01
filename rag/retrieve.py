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


def retrieve(query, k=5, n_info=None):
    """case와 info를 따로 검색해 합친 k개 청크를 유사도 순으로 반환한다.

    1인칭 고민 문장은 내담자 발화로 된 case에 쏠리므로 info 개수를 고정한다.
    n_info를 지정하면 기본 배분 대신 그 개수만큼 info를 반환한다.
    """
    n_info = default_n_info(k) if n_info is None else n_info
    collection = get_client().get_collection(COLLECTION)
    embedding = embed([query])
    results = search(collection, embedding, "case", k - n_info) + search(collection, embedding, "info", n_info)
    return sorted(results, key=lambda r: r["score"], reverse=True)
