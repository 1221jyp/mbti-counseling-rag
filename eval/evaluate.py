"""questions.jsonl로 Recall@3, Recall@5를 계산한다.

Recall@k: 정답 청크가 상위 k개에 하나라도 있는 질문의 비율 (질문당 0 또는 1).
커버리지@k: 질문별 정답 청크 중 상위 k개에 든 비율의 평균. 정답 청크가 많으면 낮게 나온다.
유형별 Recall@k: 해당 유형의 정답이 지정된 질문만 대상으로 그 유형의 정답 적중 비율을 잰다.
retrieve는 k별로 배분이 달라 k마다 따로 호출한다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import retrieve

QUESTIONS = ROOT / "eval/questions.jsonl"
KS = (3, 5)
TYPES = ("case", "info")


def main():
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    hits = {k: 0 for k in KS}
    coverage = {k: 0.0 for k in KS}
    type_hits = {(k, t): 0 for k in KS for t in TYPES}
    type_totals = {t: 0 for t in TYPES}

    for q in questions:
        relevant = set(q["relevant_chunk_ids"])
        for t in TYPES:
            type_totals[t] += any(id_.startswith(t) for id_ in relevant)
        for k in KS:
            results = retrieve(q["question"], k=k)
            found = relevant & {r["id"] for r in results}
            hits[k] += bool(found)
            coverage[k] += len(found) / len(relevant)
            for t in TYPES:
                type_hits[k, t] += any(id_.startswith(t) for id_ in found)
            if k == max(KS) and not found:
                print(f"[실패] {q['id']} {q['question']}")
                print(f"       정답: {q['relevant_categories'] + q['relevant_info_pages']}")
                print(f"       검색: {[r['title'] for r in results]}")

    for k in KS:
        print(f"Recall@{k}: {hits[k] / len(questions):.3f} ({hits[k]}/{len(questions)})"
              f"  커버리지@{k}: {coverage[k] / len(questions):.3f}")
        for t in TYPES:
            print(f"  {t} Recall@{k}: {type_hits[k, t] / type_totals[t]:.3f} ({type_hits[k, t]}/{type_totals[t]})")


if __name__ == "__main__":
    main()
