import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.extractor import extract_paper_info

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

text = ""

for doc in docs:
    if doc.metadata["source_file"] == "ALBERT.pdf":
        text += doc.page_content + "\n"

result = extract_paper_info(text)

print(result)
