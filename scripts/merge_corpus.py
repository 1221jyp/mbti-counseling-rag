"""case와 info 청크를 rag_corpus.jsonl로 합친다."""
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "data/processed"


def main():
    with (PROCESSED / "rag_corpus.jsonl").open("w", encoding="utf-8") as output:
        for name in ("case_chunks.jsonl", "info_chunks.jsonl"):
            lines = (PROCESSED / name).read_text(encoding="utf-8").splitlines()
            output.write("\n".join(lines) + "\n")
            print(name, len(lines))


if __name__ == "__main__":
    main()
