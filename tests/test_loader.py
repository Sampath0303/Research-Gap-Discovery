from pathlib import Path

from src.pdf_loader import load_all_pdfs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

docs = load_all_pdfs(PAPERS_DIR)

print("Documents:", len(docs))