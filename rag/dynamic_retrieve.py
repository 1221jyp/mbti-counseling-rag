"""Local-only dynamic retrieval. Scores are relevance signals, not safety probabilities."""
from functools import lru_cache
from time import perf_counter
import math
from rag.integrated_retrieve import (textbook_collection, existing_collection,
    PRIVATE_DB, candidates, rank, get_reranker)


@lru_cache(maxsize=1)
def case_collection():
    return existing_collection(PRIVATE_DB, 'counseling_cases')


def ranked_groups(query, *, permission_confirmed=False, candidate_k=15,
                  device='cpu', batch_size=2):
    if not permission_confirmed:
        raise PermissionError('Confirm local-use rights and review first')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Empty query')
    if candidate_k < 5 or batch_size < 1:
        raise ValueError('candidate_k >= 5 and positive batch size required')
    from rag.retrieve import embed
    textbook, cases = textbook_collection(), case_collection()
    model = get_reranker(device)
    t = perf_counter()
    vector = embed([query.strip()])
    embedding_seconds = perf_counter() - t
    t = perf_counter()
    knowledge = rank(query, candidates(textbook, vector, candidate_k, 'textbook'), model, batch_size)
    knowledge_seconds = perf_counter() - t
    t = perf_counter()
    examples = rank(query, candidates(cases, vector, candidate_k, 'case', 'counseling_case'), model, batch_size)
    case_seconds = perf_counter() - t
    return knowledge, examples, {'embedding':embedding_seconds, 'textbook_search_rerank':knowledge_seconds,
        'case_search_rerank':case_seconds, 'total':embedding_seconds+knowledge_seconds+case_seconds}


def fixed(knowledge, cases, *, k=5, case_slots=2):
    """Strict quotas, for offline comparison. Does not substitute cases for evidence."""
    if not 0 <= case_slots <= k or k < 1:
        raise ValueError('Invalid quota')
    return knowledge[:k-case_slots] + cases[:case_slots]


def dynamic(knowledge, cases, *, knowledge_threshold, case_threshold, k=5, max_cases=2):
    """Thresholds are group-specific; no cross-group score comparison.
    If no textbook evidence meets threshold, abstain rather than return cases alone.
    This is a retrieval heuristic, not a clinical or crisis classifier.
    """
    if not (k >= 1 and 0 <= max_cases < k and 0 <= knowledge_threshold <= 1 and 0 <= case_threshold <= 1):
        raise ValueError('Invalid policy')
    for r in knowledge + cases:
        s = float(r['reranker_score'])
        if not math.isfinite(s) or not 0 <= s <= 1:
            raise ValueError('Expected finite sigmoid reranker scores in [0,1]')
    eligible = [r for r in knowledge if r['reranker_score'] >= knowledge_threshold]
    if not eligible:
        return []
    examples = [r for r in cases if r['reranker_score'] >= case_threshold][:max_cases]
    return eligible[:k-len(examples)] + examples


def retrieve_dynamic(query, *, knowledge_threshold, case_threshold, permission_confirmed=False,
                     candidate_k=15, device='cpu', batch_size=2):
    """Experimental API. Do not use for real counseling before safety review."""
    knowledge, cases, _ = ranked_groups(query, permission_confirmed=permission_confirmed,
        candidate_k=candidate_k, device=device, batch_size=batch_size)
    return dynamic(knowledge, cases, knowledge_threshold=knowledge_threshold, case_threshold=case_threshold)
