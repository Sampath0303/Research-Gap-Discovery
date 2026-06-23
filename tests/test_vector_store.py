from pathlib import Path

from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PAPERS_DIR = (
    PROJECT_ROOT
    / "data"
    / "papers"
)

print(
    "PAPERS_DIR:",
    PAPERS_DIR
)

print(
    "Exists:",
    PAPERS_DIR.exists()
)

docs = load_all_pdfs(
    PAPERS_DIR
)

print(
    "Documents:",
    len(docs)
)

chunks = create_chunks(
    docs
)

print(
    "Chunks:",
    len(chunks)
)

index, embeddings, _ = (
    create_vector_store(
        chunks
    )
)

print(
    "\nTotal Chunks:",
    len(chunks)
)

print(
    "Embedding Shape:",
    embeddings.shape
)

print(
    "Vectors Stored in FAISS:",
    index.ntotal
)