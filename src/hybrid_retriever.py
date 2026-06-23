"""
Hybrid retrieval combining BM25 and FAISS for improved search results.

DEPRECATED: Use src.hybrid_search.search_hybrid() instead.
This module is maintained for backward compatibility only.
"""

from src.hybrid_search import search_hybrid as search_hybrid_new


def search_hybrid(query, index, embeddings, chunks, k=5):
    """
    Perform hybrid retrieval using both BM25 and FAISS.
    
    DEPRECATED: This function is maintained for backward compatibility.
    Use src.hybrid_search.search_hybrid() for the latest implementation
    with improved score normalization and reranking.
    
    Args:
        query: Search query string
        index: FAISS index
        embeddings: Embedding vectors
        chunks: List of document chunks
        k: Number of top results to return
    
    Returns:
        List of top k unique chunks with metadata
    """
    # Delegate to new implementation
    return search_hybrid_new(
        query=query,
        index=index,
        embeddings=embeddings,
        chunks=chunks,
        k=k,
        bm25_weight=0.5,
        faiss_weight=0.5
    )

