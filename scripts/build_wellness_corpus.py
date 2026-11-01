"""웰니스 대화 스크립트를 카테고리별 사례(case) 청크로 변환해 case_chunks.jsonl에 저장한다."""
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/wellness/웰니스_대화_스크립트_데이터셋.xlsx"
CORPUS = ROOT / "data/processed/case_chunks.jsonl"
EXCLUDED = ROOT / "data/processed/excluded_crisis_utterances.jsonl"

SOURCE = "AI Hub 웰니스 대화 스크립트 데이터셋"
SOURCE_URL = "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=267"

MIN_CHARS, MAX_CHARS = 300, 800
TARGET_CHARS = 620

# 위기 대응은 규칙 기반으로 처리하므로 검색 대상에서 제외한다.
CRISIS = {"감정/자살충동", "감정/살인욕구", "증상/자살시도", "증상/자해"}
# 상담 의도가 없는 맞장구·잡담 발화.
FILLER = {"모호함", "일반대화", "부가설명"}


def norm(text):
    return re.sub(r"[\s.,!?…~]", "", text)


def unique(items):
    seen, out = set(), []
    for item in items:
        key = norm(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def body(category, utterances, replies):
    lines = [f"상담 주제: {category.replace('/', ' > ')}"]
    lines += [f"내담자: {u}" for u in utterances]
    lines += [f"상담자: {r}" for r in replies]
    return "\n".join(lines)


def load_groups():
    df = pd.read_excel(RAW).astype({"구분": str})
    df["유저"] = df["유저"].astype(str).str.strip()
    df["구분"] = df["구분"].str.strip()

    crisis = df[df["구분"].isin(CRISIS)]
    kept = df[~df["구분"].isin(CRISIS | FILLER)]

    groups = defaultdict(lambda: {"utterances": [], "replies": []})
    for row in kept.itertuples():
        groups[row.구분]["utterances"].append(row.유저)
        if isinstance(row.챗봇, str) and row.챗봇.strip():
            groups[row.구분]["replies"].append(row.챗봇.strip())
    return groups, crisis, len(df) - len(crisis) - len(kept)


def roll_up(groups):
    """MIN_CHARS에 못 미치는 카테고리를 상위 카테고리로 합친다. 대분류까지는 올리지 않는다."""
    for depth in range(4, 1, -1):
        for category in [c for c in groups if c.count("/") + 1 == depth]:
            group = groups[category]
            size = len(body(category, unique(group["utterances"]), unique(group["replies"])))
            if size >= MIN_CHARS or category.endswith("/기타"):
                continue
            parent = category.rsplit("/", 1)[0] if depth > 2 else f"{category.split('/')[0]}/기타"
            groups[parent]["utterances"].extend(group["utterances"])
            groups[parent]["replies"].extend(group["replies"])
            del groups[category]
    return groups


def split(category, utterances, replies):
    """발화를 글자 수 기준으로 나눈다. 답변은 모든 청크에 포함해 청크 단독으로도 읽히게 한다."""
    fixed = len(body(category, [], replies))
    total = sum(len(u) + 6 for u in utterances)
    per_part = total / max(1, math.ceil(total / max(150, TARGET_CHARS - fixed)))

    chunks, current, size = [], [], 0
    for utterance in utterances:
        cost = len(utterance) + 6
        if current and (size + cost > per_part or fixed + size + cost > MAX_CHARS):
            chunks.append(current)
            current, size = [], 0
        current.append(utterance)
        size += cost
    chunks.append(current)

    # 끝에 남은 짧은 조각은 앞 청크에 붙인다.
    if len(chunks) > 1 and len(body(category, chunks[-1], replies)) < MIN_CHARS:
        if len(body(category, chunks[-2] + chunks[-1], replies)) <= MAX_CHARS:
            chunks[-2:] = [chunks[-2] + chunks[-1]]
    return [body(category, c, replies) for c in chunks]


def main():
    groups, crisis, filler_rows = load_groups()
    groups = roll_up(groups)
    today = date.today().isoformat()

    records = []
    for category in sorted(groups):
        utterances = unique(groups[category]["utterances"])
        replies = unique(groups[category]["replies"])
        prefix = hashlib.sha1(category.encode("utf-8")).hexdigest()[:8]
        for i, text in enumerate(split(category, utterances, replies)):
            records.append({
                "id": f"case-{prefix}-{i:02d}",
                "text": text,
                "title": f"웰니스 사례: {category}",
                "doc_type": "case",
                "category": category,
                "source": SOURCE,
                "source_url": SOURCE_URL,
                "collected_at": today,
            })

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with EXCLUDED.open("w", encoding="utf-8") as f:
        for row in crisis.itertuples():
            f.write(json.dumps({"category": row.구분, "utterance": row.유저}, ensure_ascii=False) + "\n")

    lengths = pd.Series([len(r["text"]) for r in records])
    print(f"제외: 위기 {len(crisis)}행, 잡담 {filler_rows}행")
    print(f"문서(카테고리) {len(groups)}개, 청크 {len(records)}개")
    print(lengths.describe().round(1).to_string())
    print(f"범위 밖: <{MIN_CHARS} {(lengths < MIN_CHARS).sum()}개, >{MAX_CHARS} {(lengths > MAX_CHARS).sum()}개")
    short = sorted({r["category"] for r in records if len(r["text"]) < MIN_CHARS})
    if short:
        print("300자 미만 카테고리:", short)


if __name__ == "__main__":
    main()
