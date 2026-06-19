import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

chunks = create_chunks(docs)

print(f"\nTotal Pages: {len(docs)}")

print(f"\nTotal Chunks: {len(chunks)}")

print("\nChunk Preview:\n")

print(chunks[0].page_content[:1000])

print("\nSource File:")

print(chunks[0].metadata["source_file"])
