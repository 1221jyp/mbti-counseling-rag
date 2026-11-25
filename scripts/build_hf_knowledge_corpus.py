"""SIR-Lab Mental_Health_Knowledge_Dataset -> 검수용 지식 RAG JSONL.

연구/교육 목적만 지원하며 승인 전 항목은 인덱싱하지 않는다.
원본/파생 데이터는 GitHub에 커밋하지 않는다.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "SIR-Lab/Mental_Health_Knowledge_Dataset"
SOURCE_URL = "https://huggingface.co/datasets/" + DATASET
RAW_FILENAME = "Knowleadge_dataset.json"  # 원본 파일명의 철자 그대로 사용
PROCESSED = ROOT / "data" / "processed"

# 질문 기반의 보수적인 키워드 필터. 임상적 적합성 분류기가 아님.
TOPICS = (
    ("stress_management", r"stress|burnout|coping|relaxation|mindfulness|meditat|diaphragmatic breathing|resilien"),
    ("emotion_regulation", r"emotion|anger|sadness|frustration|mood regulat|affect regulat|self.regulat"),
    ("cbt_psychoeducation", r"cognitive.behavio|cognitive restructur|cognitive distortion|behavioral activat|thought pattern|learned helplessness|automatic thought|problem.solving therap"),
    ("relationships", r"social support|interpersonal|relationship|assertiv|conflict|communicat|social connect|social anxi"),
    ("anxiety_depression_sleep", r"anxiety|anxious|depression|depressive|insomnia|sleep|panic|worry|ruminat"),
    ("self_esteem_acceptance", r"self.esteem|self.accept|self.compassion|self.worth|self.critici|self.efficacy|perfectionis"),
    ("professional_help", r"seek help|professional help|psychotherap|counseling|counselling|therapist|mental health service|help.seeking|treatment approach"),
)
COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in TOPICS]

# 위험 항목은 전문가 검토 전까지 검색 근거에서 제외. 완전한 안전 필터는 아님.
HIGH_RISK = re.compile(
    r"suicid|self.harm|self.injur|kill myself|overdose|drug dose|medication dos|prescri|"
    r"emergency intervention|electroconvulsive|involuntary treatment", re.I
)


def iter_qa_pairs(node):
    """1,459개 대화 안의 중첩된 seeker/supporter Q&A 쌍을 순회한다."""
    if isinstance(node, dict):
        question, answer = node.get("seeker"), node.get("supporter")
        if isinstance(question, str) and isinstance(answer, str):
            yield question, answer
        else:
            for value in node.values():
                if isinstance(value, (list, dict)):
                    yield from iter_qa_pairs(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_qa_pairs(value)


def normalize(text):
    return " ".join(text.split()).strip()


def classify(question):
    return next((name for name, regex in COMPILED if regex.search(question)), None)


def make_records(payload, min_answer_chars=65):
    stats = Counter()
    records, seen = [], set()
    for question, answer in iter_qa_pairs(payload):
        stats["qa_seen"] += 1
        question, answer = normalize(question), normalize(answer)
        if len(question) < 12 or len(answer) < min_answer_chars:
            stats["too_short"] += 1
            continue
        if HIGH_RISK.search(question + " " + answer):
            stats["high_risk_excluded"] += 1
            continue
        category = classify(question)
        if category is None:
            stats["off_topic"] += 1
            continue
        signature = re.sub(r"\W+", "", (question + "\u241e" + answer).casefold())
        if signature in seen:
            stats["duplicate"] += 1
            continue
        seen.add(signature)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
        records.append({
            "id": "hf-mhk-" + digest,
            "text": f"Question: {question}\nAnswer: {answer}",
            "title": question[:180],
            "doc_type": "info",
            "category": "hf_knowledge/" + category,
            "source": DATASET,
            "source_url": SOURCE_URL,
            "collected_at": datetime.now(timezone.utc).date().isoformat(),
            "license": "CC BY-NC-SA 4.0",
            "review_status": "candidate",
        })
        stats["selected"] += 1
        stats[category] += 1
    return records, stats


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="로컬 Knowleadge_dataset.json")
    src.add_argument("--download", action="store_true", help="Hugging Face 캐시에 다운로드")
    parser.add_argument("--output", type=Path, default=PROCESSED / "hf_knowledge_candidates.jsonl")
    parser.add_argument("--approved-ids", type=Path, help="검수한 항목의 ID 목록")
    parser.add_argument("--min-answer-chars", type=int, default=65)
    args = parser.parse_args()
    if args.min_answer_chars < 1:
        parser.error("--min-answer-chars must be positive")

    if args.download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub first: python -m pip install huggingface_hub") from exc
        input_path = Path(hf_hub_download(repo_id=DATASET, filename=RAW_FILENAME, repo_type="dataset"))
    else:
        input_path = args.input

    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    records, stats = make_records(payload, args.min_answer_chars)
    if args.approved_ids:
        approved = {
            line.strip() for line in args.approved_ids.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        records = [dict(record, review_status="approved") for record in records if record["id"] in approved]
        if not records:
            raise SystemExit("No approved records matched; check the candidate IDs.")
    elif args.output.name == "hf_knowledge_approved.jsonl":
        raise SystemExit("Approved output requires --approved-ids")

    write_jsonl(args.output, records)
    print("Source:", input_path)
    print("QA statistics:", dict(stats))
    print("Written:", len(records), args.output)
    if not args.approved_ids:
        print("CANDIDATES ONLY: manual review is required before indexing.")


if __name__ == "__main__":
    main()
