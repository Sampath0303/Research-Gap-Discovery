from tests import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store
from src.retriever import search_documents

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

print("Docs:", len(docs))

chunks = create_chunks(docs)

print("Chunks:", len(chunks))

index, embeddings, chunks = create_vector_store(chunks)

print("Index size:", index.ntotal)

results = search_documents(
    "What are limitations of transformers?",
    index,
    chunks,
    k=5
)

print("Results:", len(results))

for r in results:
    print(r.page_content[:100])
    print("-----")
