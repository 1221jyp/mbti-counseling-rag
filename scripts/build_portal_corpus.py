"""포털 원본 HTML에서 본문만 추출해 정보(info) 청크를 만든다. 원문 표현은 고치지 않는다."""
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/portal"
OUTPUT = ROOT / "data/processed/info_chunks.jsonl"

SOURCE = "국가정신건강정보포털 (국립정신건강센터)"
MIN_CHARS, MAX_CHARS = 300, 800
# 연락처는 규칙 기반 파일로만 안내하므로 기관 안내와 참고문헌은 뺀다.
SKIP_SECTIONS = {"참고문헌", "지원체계"}


def norm(text):
    return re.sub(r"\s", "", text)


def lines_of(element):
    return [line.strip() for line in element.get_text("\n").split("\n") if line.strip()]


def disease_sections(soup):
    for item in soup.select(".accordi.disease_info > li"):
        name = item.select_one(".tit").get_text(strip=True)
        content = item.select_one(".accordi_con")
        if name not in SKIP_SECTIONS and content:
            yield name, lines_of(content)


def column_sections(soup):
    content = soup.select_one(".board_con .case_txt") or soup.select_one(".board_con")
    yield "칼럼", lines_of(content)


def split_long(line):
    """MAX_CHARS보다 긴 줄을 문장 경계에서 나눈다."""
    if len(line) <= MAX_CHARS - 100:
        return [line]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", line):
        if current and len(current) + len(sentence) > MAX_CHARS - 100:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    return pieces + [current]


def chunk_page(title, sections, seen):
    """소제목 경계를 우선해 묶되, 300자에 못 미치면 다음 소제목과 합친다."""
    chunks, names, lines = [], [], []

    def render(chunk_names, chunk_lines):
        return f"{title} > {', '.join(chunk_names)}\n" + "\n".join(chunk_lines)

    def size(extra=""):
        return len(render(names, lines)) + len(extra) + 1

    def flush():
        nonlocal names, lines
        if lines:
            chunks.append((names, lines))
        names, lines = [], []

    for name, section_lines in sections:
        fresh = []
        for line in section_lines:
            key = norm(line)
            if len(key) > 40 and key in seen:
                continue
            seen.add(key)
            fresh.extend(split_long(line))
        if not fresh:
            continue
        if lines and size() >= MIN_CHARS:
            flush()
        names.append(name)
        for line in fresh:
            # 원문 줄바꿈이 문장 중간에 있어, 여유가 있으면 문장이 끝난 줄에서 끊는다.
            sentence_end = lines and lines[-1].endswith((".", "다", "요"))
            if lines and (size(line) > MAX_CHARS or (sentence_end and size(line) > MAX_CHARS - 150)):
                current = names[-1]
                flush()
                names = [current]
            lines.append(line)
    flush()

    # 마지막 조각이 짧으면 합치거나, 앞 청크의 끝 줄을 옮겨 채운다.
    if len(chunks) > 1 and len(render(*chunks[-1])) < MIN_CHARS:
        prev, last = chunks[-2], chunks[-1]
        if len(render(prev[0] + last[0], prev[1] + last[1])) <= MAX_CHARS:
            chunks[-2:] = [(list(dict.fromkeys(prev[0] + last[0])), prev[1] + last[1])]
        else:
            while len(render(*last)) < MIN_CHARS and len(render(prev[0], prev[1][:-1])) >= MIN_CHARS:
                last[1].insert(0, prev[1].pop())
    return [render(*chunk) for chunk in chunks]


def main():
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    seen, records, skipped = set(), [], []

    for page in manifest:
        # 공공누리 표시가 없는 자료는 저작권 정책상 사전 협의 대상이다.
        if not page["kogl_types"]:
            skipped.append(page["title"])
            continue
        soup = BeautifulSoup((RAW / page["file"]).read_text(encoding="utf-8"), "lxml")
        sections = disease_sections(soup) if page["kind"] == "disease" else column_sections(soup)
        prefix = hashlib.sha1(page["file"].encode("utf-8")).hexdigest()[:8]
        for i, text in enumerate(chunk_page(page["title"], sections, seen)):
            records.append({
                "id": f"info-{prefix}-{i:02d}",
                "text": text,
                "title": page["title"],
                "doc_type": "info",
                "category": page["category"],
                "source": SOURCE,
                "source_url": page["url"],
                "collected_at": page["collected_at"],
            })

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(r["text"]) for r in records]
    print(f"문서 {len({r['title'] for r in records})}개, 청크 {len(records)}개, "
          f"{min(lengths)}~{max(lengths)}자, 평균 {sum(lengths) / len(lengths):.0f}자")
    print(f"범위 밖: <{MIN_CHARS} {sum(l < MIN_CHARS for l in lengths)}개, >{MAX_CHARS} {sum(l > MAX_CHARS for l in lengths)}개")
    print("공공누리 표시 없어 제외:", skipped)
    phones = [r["id"] for r in records if re.search(r"\b(1393|129|1577-0199|109)\b", r["text"])]
    print("전화번호 포함 청크:", phones)


if __name__ == "__main__":
    main()
