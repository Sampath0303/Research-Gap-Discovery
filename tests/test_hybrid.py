"""Test hybrid retrieval functionality."""

from pathlib import Path

from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.vector_store import create_vector_store, load_vector_store
from src.hybrid_retriever import search_hybrid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


if __name__ == "__main__":
    print("Testing hybrid retrieval...")
    
    # Load documents
    docs = load_all_pdfs(PAPERS_DIR)
    print(f"Documents loaded: {len(docs)}")
    
    # Create chunks
    chunks = create_chunks(docs)
    print(f"Chunks created: {len(chunks)}")
    
    # Load or create FAISS index
    index, embeddings, stored_chunks = load_vector_store()
    
    if index is None:
        print("Creating new FAISS index...")
        index, embeddings, _ = create_vector_store(chunks)
    else:
        print("Using saved FAISS index")
        chunks = stored_chunks
    
    # Test hybrid search
    query = "transformer training challenges"
    results = search_hybrid(query, index, embeddings, chunks, k=5)
    
    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}")
    
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Paper: {result['paper']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Content preview: {result['content'][:150]}...")
    
    print("\n✓ Hybrid retrieval test completed")
