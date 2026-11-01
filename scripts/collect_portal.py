"""국가정신건강정보포털의 정보 페이지 원본 HTML을 수집한다. 요청 간격은 1초 이상이다."""
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/portal"
BASE = "https://www.mentalhealth.go.kr/portal"
USER_AGENT = "Mozilla/5.0 (student RAG research project)"
DELAY_SECONDS = 1.2

# (dissId, 페이지 제목, 코퍼스 category). 자살과 자살예방(19)은 위기 규칙으로 처리하므로 제외한다.
DISEASES = [
    (38, "우울과 우울장애", "우울"),
    (36, "주요우울장애", "우울"),
    (35, "불안장애", "불안"),
    (34, "범불안장애", "불안"),
    (68, "사회 불안장애", "불안"),
    (18, "강박장애", "불안"),
    (33, "공황장애", "공황"),
    (30, "스트레스", "스트레스"),
    (29, "적응장애", "스트레스"),
    (28, "외상후 스트레스장애", "스트레스"),
    (31, "불면장애", "수면"),
    (32, "수면과 수면장애", "수면"),
    (10, "성격장애", "대인관계"),
    (9, "경계성 성격장애", "대인관계"),
    (17, "신경성 폭식증", "식이"),
    (73, "거식증", "식이"),
    (15, "성인ADHD", "주의력"),
    (20, "알코올사용장애", "중독"),
]

# 칼럼 게시판(BBSINIT_2)의 (nttId, 제목, category).
COLUMNS = [
    (552, "애착이해를 통해 관계 회복하기", "대인관계"),
    (318, "코로나우울과 가족 갈등", "대인관계"),
    (604, "일상에서 정신건강을 지키는 지혜", "대처법"),
    (607, "일상에서 정신 건강을 지키는 지혜 '감사하기'", "대처법"),
    (620, "일상에서 정신건강을 지키는 지혜 '트라우마 극복'", "대처법"),
    (520, "휴식의 소중함", "대처법"),
    (302, "정신과 치료로써의 명상", "대처법"),
    (521, "마음이 아픈 것은 의지의 문제가 아니다", "대처법"),
    (258, "피할 수 없는 경쟁 속에서 덜 지치려면", "스트레스"),
    (479, "스트레스가 우리에게 가져다 주는 것들", "스트레스"),
    (250, "스트레스, 해마를 잘 지켜야 하는 이유", "스트레스"),
    (498, "내 친구 불안을 소개합니다.", "불안"),
    (251, "욱하는 나 좀 말려줘요... 분노조절장애", "분노"),
]


def targets():
    for diss_id, title, category in DISEASES:
        url = f"{BASE}/disease/diseaseDetail.do?dissId={diss_id}&srCodeNm={quote(title)}"
        yield f"disease_{diss_id}", "disease", title, category, url
    for ntt_id, title, category in COLUMNS:
        url = f"{BASE}/bbs/bbsDetail.do?bbsId=BBSINIT_2&nttId={ntt_id}"
        yield f"column_{ntt_id}", "column", title, category, url


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, kind, title, category, url in targets():
        path = RAW / f"{name}.html"
        if not path.exists():
            time.sleep(DELAY_SECONDS)
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
                path.write_bytes(response.read())
        html = path.read_text(encoding="utf-8")
        kogl = sorted(set(re.findall(r"img_opentype0(\d)", html)))
        manifest.append({
            "file": path.name, "kind": kind, "title": title, "category": category,
            "url": url, "kogl_types": kogl, "collected_at": date.today().isoformat(),
        })
        print(name, title, f"{len(html)}자", f"공공누리 {kogl or '표시 없음'}")
    (RAW / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
