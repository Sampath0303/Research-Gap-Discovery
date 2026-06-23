from tests import _bootstrap
from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store
from src.retriever import search_documents

docs = load_all_pdfs(_bootstrap.PAPERS_DIR)

chunks = create_chunks(docs)

index, embeddings, chunks = create_vector_store(
    chunks
)

query = "What is replaced token detection?"

results = search_documents(
    query,
    index,
    chunks,
    k=3
)

for i, result in enumerate(results):

    print(f"\n{'=' * 60}")
    print(f"Result {i+1}")
    print(f"{'=' * 60}")

    print(f"Paper : {result['paper']}")
    print(f"Page  : {result['page']}")
    print(f"Score : {result['score']:.4f}")

    print("\nContent:\n")
    print(result["content"][:500])
