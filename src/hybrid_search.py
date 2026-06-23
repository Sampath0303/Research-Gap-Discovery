"""
Hybrid Search combining BM25 (lexical) and FAISS (semantic) retrieval.

Pipeline:
  query → BM25 retrieval → FAISS retrieval → merge → deduplicate → rerank → top-k

Features:
  - Score normalization (0-1 range)
  - Intelligent deduplication by content similarity
  - Multi-factor reranking combining both scores
  - Configurable weights for BM25 vs FAISS preference
  - Preserves result structure: content, paper, page, score
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.bm25_retriever import search_bm25
from src.retriever import search_documents


class HybridSearcher:
    """
    Hybrid search engine combining BM25 and FAISS retrieval.
    
    Attributes:
        bm25_weight: Weight for BM25 score (0-1), default 0.5
        faiss_weight: Weight for FAISS score (0-1), default 0.5
        dedup_threshold: Similarity threshold for deduplication (0-1)
    """
    
    def __init__(
        self,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
        dedup_threshold: float = 0.7
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            bm25_weight: Weight for BM25 scores in reranking (0-1)
            faiss_weight: Weight for FAISS scores in reranking (0-1)
            dedup_threshold: Content similarity threshold for deduplication (0-1)
        
        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        if not (0.9 <= bm25_weight + faiss_weight <= 1.1):
            raise ValueError(
                f"Weights must sum to 1.0, got {bm25_weight + faiss_weight}"
            )
        
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.dedup_threshold = dedup_threshold
    
    def search(
        self,
        query: str,
        index,
        embeddings,
        chunks: List,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search on query.
        
        Pipeline:
          1. Retrieve from BM25 (lexical relevance)
          2. Retrieve from FAISS (semantic relevance)
          3. Merge and deduplicate results
          4. Normalize scores to 0-1 range
          5. Rerank using combined score
          6. Return top-k results
        
        Args:
            query: Search query string
            index: FAISS index object
            embeddings: Embedding vectors (used by FAISS)
            chunks: List of document chunks with metadata
            k: Number of top results to return (default 5)
        
        Returns:
            List of top-k results, each with:
              - content: Chunk text
              - paper: Source paper name
              - page: Page number
              - score: Combined hybrid score (0-1)
        """
        # Stage 1: Retrieve from both retrievers
        bm25_results = search_bm25(query, chunks, k=min(k * 3, 20))
        faiss_results = search_documents(query, index, chunks, k=min(k * 3, 20))
        
        if not bm25_results and not faiss_results:
            return []
        
        # Stage 2: Merge results with source tracking
        merged = self._merge_results(bm25_results, faiss_results)
        
        # Stage 3: Deduplicate by content similarity
        deduplicated = self._deduplicate(merged)
        
        # Stage 4: Normalize scores
        normalized = self._normalize_scores(deduplicated)
        
        # Stage 5: Rerank using combined score
        reranked = self._rerank(normalized)
        
        # Stage 6: Return top-k
        top_k = reranked[:k]
        
        # Remove internal fields before returning
        for result in top_k:
            result.pop('_bm25_score', None)
            result.pop('_faiss_score', None)
            result.pop('_combined_score', None)
            result.pop('_content_hash', None)
        
        return top_k
    
    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        faiss_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge results from both retrievers.
        
        Creates a unified result dictionary keyed by content hash,
        tracking scores from both sources.
        
        Args:
            bm25_results: Results from BM25 retriever
            faiss_results: Results from FAISS retriever
        
        Returns:
            Dictionary mapping content_hash → result with both scores
        """
        merged = {}
        
        # Add BM25 results
        for result in bm25_results:
            content_hash = self._hash_content(result['content'])
            merged[content_hash] = {
                **result,
                '_content_hash': content_hash,
                '_bm25_score': result['score'],
                '_faiss_score': None
            }
        
        # Add/merge FAISS results
        for result in faiss_results:
            content_hash = self._hash_content(result['content'])
            
            if content_hash in merged:
                # Update FAISS score for existing result
                merged[content_hash]['_faiss_score'] = result['score']
            else:
                # New result from FAISS only
                merged[content_hash] = {
                    **result,
                    '_content_hash': content_hash,
                    '_bm25_score': None,
                    '_faiss_score': result['score']
                }
        
        return merged
    
    def _deduplicate(
        self,
        merged: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Remove near-duplicate results based on content similarity.
        
        Uses string similarity to identify duplicates/near-duplicates
        and keeps the one with better combined score.
        
        Args:
            merged: Merged results dictionary
        
        Returns:
            Deduplicated results dictionary
        """
        results_list = list(merged.values())
        
        if len(results_list) <= 1:
            return merged
        
        # Mark duplicates for removal
        to_remove = set()
        
        for i in range(len(results_list)):
            if i in to_remove:
                continue
            
            content_i = results_list[i]['content']
            
            for j in range(i + 1, len(results_list)):
                if j in to_remove:
                    continue
                
                content_j = results_list[j]['content']
                
                # Calculate Jaccard similarity of tokens
                tokens_i = set(content_i.lower().split())
                tokens_j = set(content_j.lower().split())
                
                if tokens_i and tokens_j:
                    intersection = len(tokens_i & tokens_j)
                    union = len(tokens_i | tokens_j)
                    similarity = intersection / union if union > 0 else 0
                    
                    # If similar, mark lower-scored result for removal
                    if similarity >= self.dedup_threshold:
                        hash_i = results_list[i]['_content_hash']
                        hash_j = results_list[j]['_content_hash']
                        
                        # Keep the one with non-None scores (appeared in more retrievers)
                        score_count_i = sum([
                            results_list[i]['_bm25_score'] is not None,
                            results_list[i]['_faiss_score'] is not None
                        ])
                        score_count_j = sum([
                            results_list[j]['_bm25_score'] is not None,
                            results_list[j]['_faiss_score'] is not None
                        ])
                        
                        if score_count_i >= score_count_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
        
        # Remove duplicates
        for i in sorted(to_remove, reverse=True):
            del results_list[i]
        
        # Reconstruct dictionary
        result = {}
        for item in results_list:
            result[item['_content_hash']] = item
        
        return result
    
    def _normalize_scores(
        self,
        merged: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize BM25 and FAISS scores to 0-1 range.
        
        BM25 scores are inherently different from FAISS distances,
        so we normalize each independently before combining.
        
        Args:
            merged: Merged and deduplicated results (dict)
        
        Returns:
            List of results with normalized scores
        """
        # Handle both dict and list inputs
        if isinstance(merged, dict):
            results = list(merged.values())
        else:
            results = merged
        
        if not results:
            return []
        
        # Separate BM25 and FAISS scores
        bm25_scores = [
            r['_bm25_score'] for r in results 
            if r['_bm25_score'] is not None
        ]
        faiss_scores = [
            r['_faiss_score'] for r in results 
            if r['_faiss_score'] is not None
        ]
        
        # Calculate normalization ranges
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        
        # FAISS scores are distances (lower is better)
        # Normalize by max distance (worst case)
        faiss_scores_all = [
            r['_faiss_score'] for r in results 
            if r['_faiss_score'] is not None
        ]
        faiss_max = max(faiss_scores_all) if faiss_scores_all else 1.0
        
        # Normalize each result
        for result in results:
            # Normalize BM25 (higher is better)
            if result['_bm25_score'] is not None:
                bm25_norm = (
                    (result['_bm25_score'] - bm25_min) / bm25_range
                    if bm25_range > 0 else 0.0
                )
            else:
                bm25_norm = 0.5  # Neutral if not retrieved by BM25
            
            # Normalize FAISS (invert distance: lower distance = higher score)
            if result['_faiss_score'] is not None:
                faiss_norm = 1.0 - (result['_faiss_score'] / faiss_max if faiss_max > 0 else 0.0)
            else:
                faiss_norm = 0.5  # Neutral if not retrieved by FAISS
            
            result['_bm25_norm'] = bm25_norm
            result['_faiss_norm'] = faiss_norm
        
        return results
    
    def _rerank(
        self,
        normalized: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using combined weighted score.
        
        Formula: combined_score = (bm25_weight * bm25_norm) + (faiss_weight * faiss_norm)
        
        Args:
            normalized: Results with normalized scores
        
        Returns:
            Results sorted by combined score (descending)
        """
        # Calculate combined score
        for result in normalized:
            combined = (
                self.bm25_weight * result['_bm25_norm']
                + self.faiss_weight * result['_faiss_norm']
            )
            result['_combined_score'] = combined
            # Use combined score as the returned score
            result['score'] = combined
        
        # Sort by combined score (descending)
        normalized.sort(key=lambda x: x['_combined_score'], reverse=True)
        
        return normalized
    
    @staticmethod
    def _hash_content(content: str) -> int:
        """
        Create a hash of content for deduplication.
        
        Args:
            content: Text content to hash
        
        Returns:
            Integer hash of content
        """
        return hash(content.strip().lower())


def search_hybrid(
    query: str,
    index,
    embeddings,
    chunks: List,
    k: int = 5,
    bm25_weight: float = 0.5,
    faiss_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search using BM25 and FAISS.
    
    This is the main entry point function, maintaining API compatibility
    with the existing hybrid_retriever.search_hybrid function.
    
    Pipeline:
      1. BM25 retrieval (lexical/keyword)
      2. FAISS retrieval (semantic/embedding)
      3. Merge results
      4. Deduplicate by content
      5. Normalize scores (0-1 range)
      6. Rerank with combined score
      7. Return top-k
    
    Args:
        query: Search query string
        index: FAISS index object
        embeddings: Embedding vectors for FAISS
        chunks: List of document chunks
        k: Number of top results to return (default 5)
        bm25_weight: Weight for BM25 scores (default 0.5)
        faiss_weight: Weight for FAISS scores (default 0.5)
    
    Returns:
        List of top-k results with: content, paper, page, score
        Score is combined hybrid score (0-1 range, higher is better)
    
    Example:
        >>> results = search_hybrid(
        ...     query="transformer attention mechanism",
        ...     index=faiss_index,
        ...     embeddings=embeddings,
        ...     chunks=document_chunks,
        ...     k=5
        ... )
        >>> for result in results:
        ...     print(f"{result['paper']}: {result['score']:.2f}")
    """
    searcher = HybridSearcher(
        bm25_weight=bm25_weight,
        faiss_weight=faiss_weight
    )
    
    return searcher.search(query, index, embeddings, chunks, k=k)
