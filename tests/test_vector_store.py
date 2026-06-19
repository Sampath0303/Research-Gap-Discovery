import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

chunks = create_chunks(docs)

index, embeddings, _ = create_vector_store(chunks)

print("\nTotal Chunks:", len(chunks))

print("Embedding Shape:", embeddings.shape)

print("Vectors Stored in FAISS:", index.ntotal)
