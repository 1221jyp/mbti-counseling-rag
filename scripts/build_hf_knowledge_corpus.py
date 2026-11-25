"""원문 근거와 정확히 대조한 소수의 심리교육 Q&A만 인덱싱 후보로 내보낸다.

이 스크립트의 'source_crosschecked'는 교재 본문과 일반 개념을 대조했다는 의미이며,
임상적 유효성이나 개인별 치료 적합성 검증을 뜻하지 않는다.
검증 근거가 없는 나머지 Q&A는 기본적으로 모두 제외한다 (fail closed).
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATASET = "SIR-Lab/Mental_Health_Knowledge_Dataset"
SOURCE_URL = "https://huggingface.co/datasets/" + DATASET
RAW_FILENAME = "Knowleadge_dataset.json"
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MANIFEST = ROOT / "docs" / "hf_knowledge_verified_manifest.json"
OUTPUT = PROCESSED / "hf_knowledge_approved.jsonl"

TOPICS = (
    ("cbt_psychoeducation", r"cognitive.behavio|cognitive restructur|cognitive distortion|behavioral activat|automatic thought|problem.solving therap"),
    ("emotion_regulation", r"emotion|anger|sadness|frustration|mood regulat|affect regulat|self.regulat"),
    ("relationships", r"social support|interpersonal|relationship|assertiv|conflict|communicat|social connect"),
    ("self_esteem_acceptance", r"self.esteem|self.accept|self.compassion|self.worth|self.critici|self.efficacy|perfectionis"),
    ("professional_help", r"seek help|professional help|psychotherap|counseling|counselling|therapist|mental health service|help.seeking"),
    ("anxiety_depression_sleep", r"anxiety|anxious|depression|depressive|insomnia|sleep|panic|worry|ruminat"),
    ("stress_management", r"stress|burnout|coping|relaxation|mindfulness|meditat|resilien"),
)
COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in TOPICS]
HIGH_RISK = re.compile(
    r"suicid|self.harm|self.injur|kill myself|overdose|drug dose|medication dos|prescri|"
    r"emergency intervention|electroconvulsive|involuntary treatment", re.I
)
APPROVED_EVIDENCE_HOSTS = {"openstax.org", "www.openstax.org", "who.int", "www.who.int",
                           "nimh.nih.gov", "www.nimh.nih.gov", "ncbi.nlm.nih.gov"}


def iter_qa_pairs(node):
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


def stable_id(question, answer):
    signature = re.sub(r"\W+", "", (question + "\u241e" + answer).casefold())
    return "hf-mhk-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]


def make_records(payload, min_answer_chars=65):
    stats, records, seen = Counter(), [], set()
    for question, answer in iter_qa_pairs(payload):
        stats["qa_seen"] += 1
        question, answer = normalize(question), normalize(answer)
        if len(question) < 12 or len(answer) < min_answer_chars:
            stats["too_short"] += 1
            continue
        if HIGH_RISK.search(question + " " + answer):
            stats["high_risk"] += 1
            continue
        category = classify(question)
        if category is None:
            stats["off_topic"] += 1
            continue
        uid = stable_id(question, answer)
        if uid in seen:
            stats["duplicate"] += 1
            continue
        seen.add(uid)
        records.append({
            "id": uid,
            "text": f"Question: {question}\nAnswer: {answer}",
            "title": question[:180],
            "doc_type": "info",
            "category": "hf_knowledge/" + category,
            "source": DATASET,
            "source_url": SOURCE_URL,
            "collected_at": datetime.now(timezone.utc).date().isoformat(),
            "license": "CC BY-NC-SA 4.0",
            "review_status": "unverified",
        })
        stats["candidate"] += 1
    return records, stats


def read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET:
        raise ValueError("Evidence manifest does not match the HF dataset")
    entries = {}
    for entry in manifest.get("records", []):
        uid = entry.get("id")
        url = entry.get("reference_url", "")
        parsed = urlparse(url)
        if (not uid or uid in entries or parsed.scheme != "https"
                or parsed.hostname not in APPROVED_EVIDENCE_HOSTS
                or not entry.get("reference_section") or not entry.get("note")):
            raise ValueError(f"Invalid / duplicate source verification entry: {uid}")
        entries[uid] = entry
    if not entries:
        raise ValueError("No reference-checked entries in evidence manifest")
    return manifest, entries


def select_source_crosschecked(records, manifest, entries):
    """내용 해시가 근거 목록과 맞는 Q&A만 통과. 키워드만 맞는 나머지는 모두 제거."""
    chosen = []
    for record in records:
        evidence = entries.get(record["id"])
        if evidence is None:
            continue
        item = dict(record)
        item.update({
            "review_status": "source_crosschecked",
            "verification_level": "concept_source_correlated_not_clinically_validated",
            "evidence_url": evidence["reference_url"],
            "evidence_section": evidence["reference_section"],
            "evidence_note": evidence["note"],
            "evidence_checked_at": manifest["screening_date"],
            "text_sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
        })
        chosen.append(item)
    return chosen


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="local JSON path")
    src.add_argument("--download", action="store_true", help="download HF source JSON")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-answer-chars", type=int, default=65)
    args = parser.parse_args()
    if args.min_answer_chars < 1:
        parser.error("--min-answer-chars must be positive")
    if args.download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("pip install huggingface_hub") from exc
        input_path = Path(hf_hub_download(repo_id=DATASET, filename=RAW_FILENAME, repo_type="dataset"))
    else:
        input_path = args.input

    manifest, entries = read_manifest(args.manifest)
    source = json.loads(input_path.read_text(encoding="utf-8-sig"))
    candidates, stats = make_records(source, args.min_answer_chars)
    selected = select_source_crosschecked(candidates, manifest, entries)
    if not selected:
        raise SystemExit("No source-crosschecked Q&A matched the original text exactly. Output NOT modified.")
    write_jsonl(args.output, selected)
    print(f"Q&A seen: {stats['qa_seen']}; topic/quality candidates: {len(candidates)}")
    print(f"Source-crosschecked: {len(selected)} / {len(entries)} manifest entries")
    print(f"Removed from embedding: {stats['qa_seen'] - len(selected)}")
    missing = set(entries) - {item["id"] for item in selected}
    if missing:
        print(f"Manifest entries not found in this source revision: {len(missing)}")
    print("Approved corpus:", args.output)
    print("NOTE: conceptual cross-check is NOT clinical validation.")


if __name__ == "__main__":
    main()
