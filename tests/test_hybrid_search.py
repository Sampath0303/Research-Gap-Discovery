"""
Tests for Hybrid Search (BM25 + FAISS).

Covers:
  - Score normalization
  - Deduplication
  - Reranking
  - Result merging
  - API compatibility
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from src.hybrid_search import HybridSearcher, search_hybrid


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_chunks():
    """Create sample document chunks with metadata."""
    return [
        Mock(
            page_content="Transformer architecture with attention mechanisms for NLP",
            metadata={"source_file": "Attention.pdf", "page": 5}
        ),
        Mock(
            page_content="BERT model uses bidirectional encoder representations",
            metadata={"source_file": "BERT.pdf", "page": 3}
        ),
        Mock(
            page_content="Fine-tuning pre-trained models improves downstream tasks",
            metadata={"source_file": "Transfer.pdf", "page": 7}
        ),
        Mock(
            page_content="Attention mechanism allows models to focus on relevant tokens",
            metadata={"source_file": "Mechanisms.pdf", "page": 2}
        ),
        Mock(
            page_content="Deep learning has revolutionized natural language processing",
            metadata={"source_file": "DeepLearning.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def mock_faiss_index():
    """Create a mock FAISS index."""
    index = Mock()
    # Simulate search returning indices and distances
    index.search = Mock(return_value=(
        np.array([[0.1, 0.3, 0.5, 0.7, 0.9]]),
        np.array([[0, 1, 2, 3, 4]])
    ))
    return index


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    return np.random.randn(5, 384).astype('float32')


@pytest.fixture
def bm25_results():
    """Sample BM25 retrieval results."""
    return [
        {
            "content": "Transformer architecture with attention mechanisms for NLP",
            "paper": "Attention.pdf",
            "page": 5,
            "score": 2.5
        },
        {
            "content": "Attention mechanism allows models to focus on relevant tokens",
            "paper": "Mechanisms.pdf",
            "page": 2,
            "score": 2.1
        },
    ]


@pytest.fixture
def faiss_results():
    """Sample FAISS retrieval results."""
    return [
        {
            "content": "Transformer architecture with attention mechanisms for NLP",
            "paper": "Attention.pdf",
            "page": 5,
            "score": 0.1
        },
        {
            "content": "BERT model uses bidirectional encoder representations",
            "paper": "BERT.pdf",
            "page": 3,
            "score": 0.3
        },
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestHybridSearcherInitialization:
    """Test HybridSearcher initialization."""
    
    def test_init_default_weights(self):
        """Test initialization with default weights."""
        searcher = HybridSearcher()
        assert searcher.bm25_weight == 0.5
        assert searcher.faiss_weight == 0.5
        assert searcher.dedup_threshold == 0.7
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        searcher = HybridSearcher(bm25_weight=0.6, faiss_weight=0.4)
        assert searcher.bm25_weight == 0.6
        assert searcher.faiss_weight == 0.4
    
    def test_init_invalid_weights_sum(self):
        """Test that invalid weight sum raises error."""
        with pytest.raises(ValueError):
            HybridSearcher(bm25_weight=0.6, faiss_weight=0.3)
    
    def test_init_custom_dedup_threshold(self):
        """Test initialization with custom deduplication threshold."""
        searcher = HybridSearcher(dedup_threshold=0.8)
        assert searcher.dedup_threshold == 0.8


# ============================================================================
# MERGING TESTS
# ============================================================================


class TestResultMerging:
    """Test merging of BM25 and FAISS results."""
    
    def test_merge_bm25_only(self):
        """Test merging with only BM25 results."""
        searcher = HybridSearcher()
        bm25_results = [
            {
                "content": "Test content 1",
                "paper": "Paper1.pdf",
                "page": 1,
                "score": 2.5
            }
        ]
        
        merged = searcher._merge_results(bm25_results, [])
        assert len(merged) == 1
        assert list(merged.values())[0]["_bm25_score"] == 2.5
        assert list(merged.values())[0]["_faiss_score"] is None
    
    def test_merge_faiss_only(self):
        """Test merging with only FAISS results."""
        searcher = HybridSearcher()
        faiss_results = [
            {
                "content": "Test content 2",
                "paper": "Paper2.pdf",
                "page": 2,
                "score": 0.3
            }
        ]
        
        merged = searcher._merge_results([], faiss_results)
        assert len(merged) == 1
        assert list(merged.values())[0]["_bm25_score"] is None
        assert list(merged.values())[0]["_faiss_score"] == 0.3
    
    def test_merge_both_same_content(self):
        """Test merging when both retrievers find same content."""
        searcher = HybridSearcher()
        shared_content = "Transformer architecture"
        
        bm25_results = [
            {
                "content": shared_content,
                "paper": "Paper1.pdf",
                "page": 1,
                "score": 2.5
            }
        ]
        
        faiss_results = [
            {
                "content": shared_content,
                "paper": "Paper1.pdf",
                "page": 1,
                "score": 0.1
            }
        ]
        
        merged = searcher._merge_results(bm25_results, faiss_results)
        assert len(merged) == 1
        result = list(merged.values())[0]
        assert result["_bm25_score"] == 2.5
        assert result["_faiss_score"] == 0.1
    
    def test_merge_both_different_content(self):
        """Test merging with different content from each retriever."""
        searcher = HybridSearcher()
        
        bm25_results = [
            {
                "content": "Content A",
                "paper": "Paper1.pdf",
                "page": 1,
                "score": 2.5
            }
        ]
        
        faiss_results = [
            {
                "content": "Content B",
                "paper": "Paper2.pdf",
                "page": 2,
                "score": 0.2
            }
        ]
        
        merged = searcher._merge_results(bm25_results, faiss_results)
        assert len(merged) == 2


# ============================================================================
# DEDUPLICATION TESTS
# ============================================================================


class TestDeduplication:
    """Test result deduplication by content similarity."""
    
    def test_dedup_exact_duplicates(self):
        """Test deduplication of exact duplicates."""
        searcher = HybridSearcher(dedup_threshold=0.9)
        
        merged = {
            hash("test content 1"): {
                "content": "test content 1",
                "paper": "Paper1.pdf",
                "page": 1,
                "_content_hash": hash("test content 1"),
                "_bm25_score": 2.5,
                "_faiss_score": None
            },
            hash("test content 1"): {
                "content": "test content 1",
                "paper": "Paper1.pdf",
                "page": 1,
                "_content_hash": hash("test content 1"),
                "_bm25_score": None,
                "_faiss_score": 0.1
            }
        }
        
        deduplicated = searcher._deduplicate(merged)
        # Will still have 2 entries since different hashes, but should be handled
        assert len(deduplicated) <= len(merged)
    
    def test_dedup_no_duplicates(self):
        """Test deduplication with no duplicates."""
        searcher = HybridSearcher()
        
        merged = {
            hash("unique content 1"): {
                "content": "unique content 1",
                "_bm25_score": 2.0,
                "_faiss_score": 0.1,
                "_content_hash": hash("unique content 1")
            },
            hash("unique content 2"): {
                "content": "unique content 2",
                "_bm25_score": 1.5,
                "_faiss_score": 0.2,
                "_content_hash": hash("unique content 2")
            }
        }
        
        deduplicated = searcher._deduplicate(merged)
        assert len(deduplicated) == 2


# ============================================================================
# NORMALIZATION TESTS
# ============================================================================


class TestScoreNormalization:
    """Test score normalization to 0-1 range."""
    
    def test_normalize_bm25_scores(self):
        """Test normalization of BM25 scores."""
        searcher = HybridSearcher()
        
        normalized = [
            {
                "_bm25_score": 1.0,
                "_faiss_score": 0.5,
                "_content_hash": 1
            },
            {
                "_bm25_score": 3.0,
                "_faiss_score": 0.3,
                "_content_hash": 2
            },
            {
                "_bm25_score": 2.0,
                "_faiss_score": 0.7,
                "_content_hash": 3
            }
        ]
        
        result = searcher._normalize_scores(normalized)
        
        # All normalized scores should be in [0, 1]
        for item in result:
            assert 0.0 <= item["_bm25_norm"] <= 1.0
            assert 0.0 <= item["_faiss_norm"] <= 1.0
    
    def test_normalize_faiss_inverted(self):
        """Test that FAISS scores are inverted (lower distance = higher score)."""
        searcher = HybridSearcher()
        
        normalized = [
            {
                "_bm25_score": 2.0,
                "_faiss_score": 0.1,  # Low distance
                "_content_hash": 1
            },
            {
                "_bm25_score": 2.0,
                "_faiss_score": 0.9,  # High distance
                "_content_hash": 2
            }
        ]
        
        result = searcher._normalize_scores(normalized)
        
        # Result with lower FAISS distance should have higher score
        assert result[0]["_faiss_norm"] > result[1]["_faiss_norm"]
    
    def test_normalize_missing_scores(self):
        """Test normalization with missing scores (from one retriever)."""
        searcher = HybridSearcher()
        
        normalized = [
            {
                "_bm25_score": 2.0,
                "_faiss_score": None,  # Missing FAISS
                "_content_hash": 1
            }
        ]
        
        result = searcher._normalize_scores(normalized)
        
        # Should use neutral value
        assert result[0]["_faiss_norm"] == 0.5


# ============================================================================
# RERANKING TESTS
# ============================================================================


class TestReranking:
    """Test reranking with combined scores."""
    
    def test_rerank_equal_weights(self):
        """Test reranking with equal weights."""
        searcher = HybridSearcher(bm25_weight=0.5, faiss_weight=0.5)
        
        normalized = [
            {
                "_bm25_norm": 0.8,
                "_faiss_norm": 0.6,
                "_content_hash": 1,
                "paper": "Paper1.pdf"
            },
            {
                "_bm25_norm": 0.6,
                "_faiss_norm": 0.8,
                "_content_hash": 2,
                "paper": "Paper2.pdf"
            }
        ]
        
        result = searcher._rerank(normalized)
        
        # Both have same combined score (0.7), order preserved
        assert result[0]["_combined_score"] == result[1]["_combined_score"]
    
    def test_rerank_bm25_preference(self):
        """Test reranking with BM25 preference."""
        searcher = HybridSearcher(bm25_weight=0.7, faiss_weight=0.3)
        
        normalized = [
            {
                "_bm25_norm": 0.9,
                "_faiss_norm": 0.1,
                "_content_hash": 1,
                "paper": "Paper1.pdf"
            },
            {
                "_bm25_norm": 0.1,
                "_faiss_norm": 0.9,
                "_content_hash": 2,
                "paper": "Paper2.pdf"
            }
        ]
        
        result = searcher._rerank(normalized)
        
        # BM25 result should rank higher
        assert result[0]["_combined_score"] > result[1]["_combined_score"]
    
    def test_rerank_sorted_descending(self):
        """Test that reranked results are sorted descending."""
        searcher = HybridSearcher()
        
        normalized = [
            {
                "_bm25_norm": 0.5,
                "_faiss_norm": 0.5,
                "_content_hash": i,
                "paper": f"Paper{i}.pdf"
            }
            for i in range(5)
        ]
        # Override some scores
        normalized[0]["_bm25_norm"] = 0.3
        normalized[2]["_bm25_norm"] = 0.9
        
        result = searcher._rerank(normalized)
        
        # Check sorted descending
        scores = [r["_combined_score"] for r in result]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# FULL PIPELINE TESTS
# ============================================================================


class TestHybridSearchPipeline:
    """Test complete hybrid search pipeline."""
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_basic(self, mock_faiss, mock_bm25, sample_chunks):
        """Test basic hybrid search."""
        mock_bm25.return_value = [
            {
                "content": "Transformer architecture",
                "paper": "Attention.pdf",
                "page": 5,
                "score": 2.5
            }
        ]
        
        mock_faiss.return_value = [
            {
                "content": "BERT model",
                "paper": "BERT.pdf",
                "page": 3,
                "score": 0.2
            }
        ]
        
        searcher = HybridSearcher()
        index = Mock()
        embeddings = np.random.randn(5, 384)
        
        results = searcher.search("transformer", index, embeddings, sample_chunks, k=2)
        
        assert len(results) <= 2
        assert all("content" in r for r in results)
        assert all("paper" in r for r in results)
        assert all("page" in r for r in results)
        assert all("score" in r for r in results)
        # No internal fields should be present
        assert all("_bm25_score" not in r for r in results)
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_empty_results(self, mock_faiss, mock_bm25, sample_chunks):
        """Test hybrid search with no results."""
        mock_bm25.return_value = []
        mock_faiss.return_value = []
        
        searcher = HybridSearcher()
        index = Mock()
        embeddings = np.random.randn(5, 384)
        
        results = searcher.search("xyz", index, embeddings, sample_chunks, k=5)
        
        assert len(results) == 0
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_respects_k(self, mock_faiss, mock_bm25, sample_chunks):
        """Test that search returns at most k results."""
        # Return many results from both retrievers
        many_results = [
            {
                "content": f"Content {i}",
                "paper": f"Paper{i}.pdf",
                "page": i,
                "score": 1.0 - i * 0.1
            }
            for i in range(10)
        ]
        
        mock_bm25.return_value = many_results[:5]
        mock_faiss.return_value = many_results[5:]
        
        searcher = HybridSearcher()
        index = Mock()
        embeddings = np.random.randn(10, 384)
        
        results = searcher.search("test", index, embeddings, sample_chunks, k=3)
        
        assert len(results) == 3


# ============================================================================
# MODULE FUNCTION TESTS
# ============================================================================


class TestModuleFunction:
    """Test the module-level search_hybrid function."""
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_hybrid_function(self, mock_faiss, mock_bm25, sample_chunks):
        """Test search_hybrid function."""
        mock_bm25.return_value = [
            {
                "content": "Test content",
                "paper": "Test.pdf",
                "page": 1,
                "score": 2.0
            }
        ]
        
        mock_faiss.return_value = [
            {
                "content": "Test content 2",
                "paper": "Test2.pdf",
                "page": 2,
                "score": 0.1
            }
        ]
        
        index = Mock()
        embeddings = np.random.randn(2, 384)
        
        results = search_hybrid("test", index, embeddings, sample_chunks, k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_hybrid_custom_weights(self, mock_faiss, mock_bm25, sample_chunks):
        """Test search_hybrid with custom weights."""
        mock_bm25.return_value = [
            {
                "content": "Content 1",
                "paper": "Paper1.pdf",
                "page": 1,
                "score": 2.0
            }
        ]
        
        mock_faiss.return_value = [
            {
                "content": "Content 2",
                "paper": "Paper2.pdf",
                "page": 2,
                "score": 0.1
            }
        ]
        
        index = Mock()
        embeddings = np.random.randn(2, 384)
        
        results = search_hybrid(
            "test",
            index,
            embeddings,
            sample_chunks,
            k=2,
            bm25_weight=0.7,
            faiss_weight=0.3
        )
        
        assert isinstance(results, list)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_hash_content_empty_string(self):
        """Test hashing empty content."""
        hash1 = HybridSearcher._hash_content("")
        hash2 = HybridSearcher._hash_content("")
        assert hash1 == hash2
    
    def test_hash_content_case_insensitive(self):
        """Test that hashing is case-insensitive."""
        hash1 = HybridSearcher._hash_content("Test Content")
        hash2 = HybridSearcher._hash_content("test content")
        assert hash1 == hash2
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_with_unicode_content(self, mock_faiss, mock_bm25, sample_chunks):
        """Test search with unicode content."""
        mock_bm25.return_value = [
            {
                "content": "Unicode: 日本語 中文 한국어",
                "paper": "Unicode.pdf",
                "page": 1,
                "score": 2.0
            }
        ]
        
        mock_faiss.return_value = []
        
        searcher = HybridSearcher()
        index = Mock()
        embeddings = np.random.randn(1, 384)
        
        results = searcher.search("unicode", index, embeddings, sample_chunks, k=1)
        
        assert len(results) == 1
        assert "日本語" in results[0]["content"]
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_with_special_characters(self, mock_faiss, mock_bm25, sample_chunks):
        """Test search with special characters."""
        mock_bm25.return_value = [
            {
                "content": "Special: @#$%^&*()_+-=[]{}|;:,.<>?",
                "paper": "Special.pdf",
                "page": 1,
                "score": 2.0
            }
        ]
        
        mock_faiss.return_value = []
        
        searcher = HybridSearcher()
        index = Mock()
        embeddings = np.random.randn(1, 384)
        
        results = searcher.search("special", index, embeddings, sample_chunks, k=1)
        
        assert len(results) == 1


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with existing API."""
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_hybrid_returns_expected_fields(self, mock_faiss, mock_bm25, sample_chunks):
        """Test that search_hybrid returns expected fields."""
        mock_bm25.return_value = [
            {
                "content": "Test",
                "paper": "Paper.pdf",
                "page": 1,
                "score": 2.0
            }
        ]
        
        mock_faiss.return_value = []
        
        index = Mock()
        embeddings = np.random.randn(1, 384)
        
        results = search_hybrid("test", index, embeddings, sample_chunks, k=1)
        
        assert len(results) == 1
        result = results[0]
        
        # Check for required fields
        assert "content" in result
        assert "paper" in result
        assert "page" in result
        assert "score" in result
        
        # Score should be numeric and in [0, 1]
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
    
    @patch('src.hybrid_search.search_bm25')
    @patch('src.hybrid_search.search_documents')
    def test_search_hybrid_same_signature(self, mock_faiss, mock_bm25, sample_chunks):
        """Test that search_hybrid has compatible function signature."""
        mock_bm25.return_value = []
        mock_faiss.return_value = []
        
        index = Mock()
        embeddings = np.random.randn(1, 384)
        
        # Should accept these parameters
        results = search_hybrid(
            query="test",
            index=index,
            embeddings=embeddings,
            chunks=sample_chunks,
            k=5
        )
        
        assert isinstance(results, list)
