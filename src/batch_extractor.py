import json
from pathlib import Path

from src.extractor import extract_paper_info
from src.pdf_loader import load_all_pdfs


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
OUTPUTS_DIR = BASE_DIR / "outputs"


def extract_all_papers():
    docs = load_all_pdfs(PAPERS_DIR)
    papers = {}

    for doc in docs:
        file_name = doc.metadata["source_file"]
        papers.setdefault(file_name, "")
        papers[file_name] += doc.page_content + "\n"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for paper_name, text in papers.items():
        output_path = OUTPUTS_DIR / f"{Path(paper_name).stem}.json"

        if output_path.exists():
            continue

        info = extract_paper_info(text)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=4)


def main():
    extract_all_papers()


if __name__ == "__main__":
    main()
