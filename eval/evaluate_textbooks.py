"""Evaluate WHO/OpenStax source-selected retrieval; draft labels, not clinical evaluation."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metrics(ids, relevant, k):
    returned = ids[:k]
    hits = sum(uid in relevant for uid in returned)
    return {
        "hit": float(hits > 0),
        "precision": hits / k,
        "recall": hits / len(relevant),
        "mrr": next((1 / i for i, uid in enumerate(returned, 1) if uid in relevant), 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/textbook_eval.json")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be positive")
    from rag.retrieve import embed
    from rag.who_knowledge import search_who
    from rag.openstax_knowledge import search_openstax
    questions = [json.loads(line) for line in (ROOT / "eval/textbook_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = defaultdict(list)
    details = []
    for q in questions:
        emb = embed([q["question"]])
        who = search_who(emb, args.k)
        ost = search_openstax(emb, args.k)
        relevant = set(q["relevant_chunk_ids"])
        ranked = {
            "who": who, "openstax": ost,
            "combined": sorted(who + ost, key=lambda r: r["score"], reverse=True),
        }
        detail = {"id": q["id"], "category": q["category"], "results": {}}
        for mode, results in ranked.items():
            gold = {uid for uid in relevant if uid.startswith("who-")} if mode == "who" else (
                {uid for uid in relevant if uid.startswith("openstax-")} if mode == "openstax" else relevant)
            ids = [r["id"] for r in results[:args.k]]
            if gold:
                m = metrics(ids, gold, args.k)
                scores[(mode, "all")].append(m)
                scores[(mode, q["category"])].append(m)
            else:
                m = None
            detail["results"][mode] = {"ids": ids, "metrics": m}
        details.append(detail)
    summary = {}
    for (mode, category), values in scores.items():
        summary.setdefault(mode, {})[category] = {
            "n_questions": len(values),
            **{key: sum(v[key] for v in values) / len(values) for key in values[0]},
        }
    output = {"k": args.k, "status": "draft_relevance_labels_not_clinical_validation",
              "summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({m: s.get("all") for m, s in summary.items()}, ensure_ascii=False, indent=2))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
