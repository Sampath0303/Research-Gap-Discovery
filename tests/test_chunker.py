from pathlib import Path

from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

print("PAPERS_DIR:", PAPERS_DIR)
print("Exists:", PAPERS_DIR.exists())

docs = load_all_pdfs(PAPERS_DIR)

chunks = create_chunks(docs)

print(f"\nTotal Pages: {len(docs)}")
print(f"\nTotal Chunks: {len(chunks)}")

if chunks:
    print("\nChunk Preview:\n")
    print(chunks[0].page_content[:1000])

    print("\nSource File:")
    print(chunks[0].metadata.get("source_file", "Unknown"))
else:
    print("\nNo chunks generated.")