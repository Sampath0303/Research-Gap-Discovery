# Hybrid Search Implementation Guide

## Overview

Hybrid Search combines **BM25** (lexical/keyword-based retrieval) and **FAISS** (semantic/embedding-based retrieval) to provide comprehensive document search with:
- 🎯 Improved relevance (catches both keyword and semantic matches)
- 🔧 Configurable weights (adjust BM25 vs FAISS preference)
- 📊 Intelligent score normalization (0-1 range)
- 🚫 Automatic deduplication
- 📈 Smart reranking with combined scoring

## Architecture

### Pipeline

```
query
  ↓
[Stage 1] BM25 Retrieval (lexical)
  ↓
[Stage 2] FAISS Retrieval (semantic)
  ↓
[Stage 3] Merge Results
  Track both BM25 and FAISS scores for each document
  ↓
[Stage 4] Deduplicate
  Remove near-duplicates using Jaccard similarity (default threshold: 0.7)
  ↓
[Stage 5] Normalize Scores
  • BM25: min-max normalization (higher is better)
  • FAISS: inverted normalization (distance → 0-1 score)
  ↓
[Stage 6] Rerank
  combined_score = (bm25_weight × bm25_norm) + (faiss_weight × faiss_norm)
  ↓
[Stage 7] Return Top-k
  Sorted by combined score (descending)
```

### Key Components

#### 1. HybridSearcher Class

Main class orchestrating the hybrid search pipeline.

```python
from src.hybrid_search import HybridSearcher

searcher = HybridSearcher(
    bm25_weight=0.5,        # Weight for BM25 scores (default 0.5)
    faiss_weight=0.5,       # Weight for FAISS scores (default 0.5)
    dedup_threshold=0.7     # Content similarity threshold (default 0.7)
)

results = searcher.search(
    query="transformer attention",
    index=faiss_index,
    embeddings=embeddings,
    chunks=document_chunks,
    k=5  # Return top-5 results
)
```

#### 2. Module Function

Simple API for common use cases.

```python
from src.hybrid_search import search_hybrid

results = search_hybrid(
    query="transformer attention",
    index=faiss_index,
    embeddings=embeddings,
    chunks=document_chunks,
    k=5,
    bm25_weight=0.5,
    faiss_weight=0.5
)
```

### Result Format

Each result is a dictionary with:
```python
{
    "content": "Document text excerpt",
    "paper": "source_paper.pdf",
    "page": 5,
    "score": 0.85  # Combined hybrid score (0-1, higher is better)
}
```

## Score Normalization

### BM25 Score Normalization

BM25 produces raw scores that vary based on query and corpus statistics.

```
Raw BM25 scores: [1.2, 3.5, 2.8]
Min: 1.2, Max: 3.5, Range: 2.3

Normalized: [(1.2-1.2)/2.3, (3.5-1.2)/2.3, (2.8-1.2)/2.3]
           = [0.0, 1.0, 0.696]
```

### FAISS Distance Normalization

FAISS returns distances (L2 in this case). Lower distance = higher similarity.

```
Raw FAISS distances: [0.1, 0.3, 0.8]
Max: 0.8

Normalized (inverted): [1.0-(0.1/0.8), 1.0-(0.3/0.8), 1.0-(0.8/0.8)]
                      = [0.875, 0.625, 0.0]
```

### Combined Score

Weighted average of normalized scores:

```
combined_score = (bm25_weight × bm25_norm) + (faiss_weight × faiss_norm)

Example with equal weights (0.5, 0.5):
Result 1: (0.5 × 0.8) + (0.5 × 0.9) = 0.85
Result 2: (0.5 × 0.2) + (0.5 × 0.8) = 0.50
```

## Configuration Examples

### Balanced Search (Default)
```python
results = search_hybrid(query, index, embeddings, chunks, 
                       bm25_weight=0.5, faiss_weight=0.5)
```
Best for: General use, balanced relevance

### Keyword-Focused Search
```python
results = search_hybrid(query, index, embeddings, chunks,
                       bm25_weight=0.7, faiss_weight=0.3)
```
Best for: Exact phrase matching, technical terms

### Semantic-Focused Search
```python
results = search_hybrid(query, index, embeddings, chunks,
                       bm25_weight=0.3, faiss_weight=0.7)
```
Best for: Concept-based search, paraphrases

## Deduplication

The system identifies and removes near-duplicate documents using Jaccard similarity of tokens.

```python
# Two very similar documents
doc1 = "Transformers revolutionized natural language processing"
doc2 = "Transformers revolutionized natural language processing tasks"

# Jaccard similarity: |intersection| / |union|
tokens1 = {"transformers", "revolutionized", "natural", "language", "processing"}
tokens2 = {"transformers", "revolutionized", "natural", "language", "processing", "tasks"}

jaccard = 5 / 6 ≈ 0.833  # > 0.7 threshold

# Result: One document is removed (kept is the one with better combined score)
```

## Integration with Dashboard

Updated in `src/dashboard_utils.py`:

```python
def perform_semantic_search(query: str, top_k: int = 5, 
                            bm25_weight: float = 0.5,
                            faiss_weight: float = 0.5):
    """Hybrid search with optional weight configuration."""
    from src.hybrid_search import search_hybrid
    
    retrieved_chunks = search_hybrid(
        query, index, embeddings, chunks,
        k=top_k,
        bm25_weight=bm25_weight,
        faiss_weight=faiss_weight
    )
    
    return retrieved_chunks, answer
```

## Backward Compatibility

The new `hybrid_search.py` maintains full API compatibility:

```python
# Old code (still works)
from src.hybrid_retriever import search_hybrid
results = search_hybrid(query, index, embeddings, chunks, k=5)

# New code (better implementation)
from src.hybrid_search import search_hybrid
results = search_hybrid(query, index, embeddings, chunks, k=5)

# Both return identical structure
```

The original `hybrid_retriever.py` now delegates to `hybrid_search.py` for backward compatibility.

## Testing

Comprehensive test suite with 27 tests covering:

```
✓ Initialization & Configuration (4 tests)
✓ Result Merging (4 tests)
✓ Deduplication (2 tests)
✓ Score Normalization (3 tests)
✓ Reranking (3 tests)
✓ Full Pipeline (3 tests)
✓ Module Functions (2 tests)
✓ Edge Cases (4 tests)
✓ Backward Compatibility (2 tests)
```

Run tests:
```bash
pytest tests/test_hybrid_search.py -v
```

## Performance

- **BM25 Retrieval**: ~10-50ms (depends on corpus size)
- **FAISS Retrieval**: ~1-10ms (vectorized operation)
- **Score Normalization**: <1ms (linear scan)
- **Reranking**: <1ms (sorting)
- **Total Pipeline**: ~15-60ms for typical queries

## Troubleshooting

### No Results Found
- Check if documents are loaded (`len(chunks) > 0`)
- Verify FAISS index is created
- Try broadening query terms

### Unexpected Scoring
- Check weight configuration
- Verify both BM25 and FAISS are contributing
- Consider adjusting deduplication threshold

### Performance Issues
- Reduce `k` parameter (fewer results to process)
- Reduce corpus size if possible
- Use pre-built index (don't rebuild each time)

## Advanced Usage

### Custom Weight Tuning

For domain-specific searches, tune weights based on your data:

```python
# For technical papers (favor exact terms)
results = search_hybrid(query, index, embeddings, chunks,
                       bm25_weight=0.7, faiss_weight=0.3)

# For creative content (favor concepts)
results = search_hybrid(query, index, embeddings, chunks,
                       bm25_weight=0.3, faiss_weight=0.7)

# For balanced requirements
results = search_hybrid(query, index, embeddings, chunks,
                       bm25_weight=0.5, faiss_weight=0.5)
```

### Deduplication Control

Adjust similarity threshold to control duplicate removal:

```python
searcher = HybridSearcher(
    bm25_weight=0.5,
    faiss_weight=0.5,
    dedup_threshold=0.8  # Stricter (remove more duplicates)
)

# vs

searcher = HybridSearcher(
    bm25_weight=0.5,
    faiss_weight=0.5,
    dedup_threshold=0.6  # Lenient (keep more variations)
)
```

## Future Enhancements

Potential improvements for consideration:
- [ ] Cross-encoder reranking for additional quality boost
- [ ] Query expansion (expand query with related terms)
- [ ] Multi-field search (search specific fields separately)
- [ ] Fusion scoring algorithms (RRF, score-based fusion)
- [ ] Adaptive weight learning (learn optimal weights from data)

## References

- **BM25**: https://en.wikipedia.org/wiki/Okapi_BM25
- **FAISS**: https://github.com/facebookresearch/faiss
- **Score Normalization**: https://en.wikipedia.org/wiki/Min-max_normalization
- **Jaccard Similarity**: https://en.wikipedia.org/wiki/Jaccard_index
