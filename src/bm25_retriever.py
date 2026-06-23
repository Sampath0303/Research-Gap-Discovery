"""BM25 retrieval for keyword-based search."""

from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    """Simple tokenization for BM25."""
    # Convert to lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\w+', text.lower())
    return tokens


def build_bm25_index(chunks):
    """Build BM25 index from chunks."""
    if not chunks:
        return None
    
    # Tokenize all chunk contents
    tokenized_corpus = [tokenize(chunk.page_content) for chunk in chunks]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def search_bm25(query, chunks, k=5):
    """Search using BM25 and return results in same format as FAISS retriever."""
    if not chunks:
        return []
    
    # Build BM25 index
    bm25 = build_bm25_index(chunks)
    if bm25 is None:
        return []
    
    # Tokenize query
    tokenized_query = tokenize(query)
    
    # Get top k results
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-k:][::-1]
    
    # Format results to match FAISS retriever format
    results = []
    for rank, idx in enumerate(top_indices):
        chunk = chunks[idx]
        results.append({
            "content": chunk.page_content,
            "paper": chunk.metadata.get("source_file", "Unknown"),
            "page": chunk.metadata.get("page", "?"),
            "score": float(doc_scores[idx])
        })
    
    return results
