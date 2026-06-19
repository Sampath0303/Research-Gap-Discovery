import _bootstrap
from src.pdf_loader import load_all_pdfs

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

print(f"\nTotal Pages Loaded: {len(docs)}")

print("\nFirst Page Preview:\n")

print(docs[0].page_content[:1000])

print("\nSource PDF:")

print(docs[0].metadata["source_file"])
