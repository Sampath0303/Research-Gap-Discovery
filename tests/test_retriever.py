import _bootstrap
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

    print(f"\nResult {i+1}")

    print(
        "Source:",
        result.metadata["source_file"]
    )

    print(
        result.page_content[:500]
    )
