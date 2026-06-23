"""Test BM25 retrieval functionality."""

from pathlib import Path

from src.pdf_loader import load_all_pdfs
from src.chunker import create_chunks
from src.bm25_retriever import search_bm25


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"


if __name__ == "__main__":
    print("Testing BM25 retrieval...")
    
    # Load documents
    docs = load_all_pdfs(PAPERS_DIR)
    print(f"Documents loaded: {len(docs)}")
    
    # Create chunks
    chunks = create_chunks(docs)
    print(f"Chunks created: {len(chunks)}")
    
    # Test BM25 search
    query = "transformer training challenges"
    results = search_bm25(query, chunks, k=5)
    
    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}")
    
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Paper: {result['paper']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content preview: {result['content'][:150]}...")
    
    print("\n✓ BM25 retrieval test completed")
