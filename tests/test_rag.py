import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store
from src.retriever import search_documents
from src.rag import generate_answer

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

chunks = create_chunks(docs)

index, embeddings, chunks = create_vector_store(
    chunks
)

query = "What is parameter sharing in ALBERT?"

retrieved = search_documents(
    query,
    index,
    chunks,
    k=5
)

answer = generate_answer(
    query,
    retrieved
)

print("\nANSWER:\n")
print(answer)
