"""
Tests for incremental indexing functionality.

Tests:
- Detecting new PDFs
- Processing individual PDFs
- Appending to FAISS index
- Registry updates
- Full incremental batch processing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import faiss
import pickle

from tests import _bootstrap
from _bootstrap import BASE_DIR
from src.incremental_indexer import IncrementalIndexer, run_incremental_indexing
from src.paper_registry import load_registry, save_registry, update_paper
from src.vector_store import load_vector_store, save_updated_index, get_index_stats


class TestIncrementalIndexer:
    """Test IncrementalIndexer class."""
    
    def test_initialization(self):
        """Test that indexer initializes properly."""
        indexer = IncrementalIndexer()
        
        assert indexer.registry is not None
        assert indexer.chunker is not None
        # Index/embeddings may be None if not yet created
        assert isinstance(indexer.registry, dict)
    
    def test_detect_new_pdfs(self):
        """Test detection of unprocessed PDFs."""
        indexer = IncrementalIndexer()
        new_pdfs = indexer.detect_new_pdfs()
        
        # Should be a list
        assert isinstance(new_pdfs, list)
        
        # All items should be Path objects
        for pdf in new_pdfs:
            assert isinstance(pdf, Path)
            assert pdf.suffix == ".pdf"
    
    def test_process_existing_pdf(self):
        """Test processing an existing PDF."""
        indexer = IncrementalIndexer()
        papers_dir = BASE_DIR / "data" / "papers"
        
        if not papers_dir.exists():
            pytest.skip("Papers directory not found")
        
        # Get first PDF
        pdfs = list(papers_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs found")
        
        pdf_path = pdfs[0]
        chunks, pages, chunk_count = indexer.process_new_pdf(pdf_path)
        
        # Should have extracted data
        assert isinstance(chunks, list)
        assert pages > 0
        assert chunk_count > 0
        assert len(chunks) == chunk_count
        assert all(isinstance(c, str) for c in chunks)
    
    def test_append_to_faiss_empty(self):
        """Test appending to FAISS with empty chunks."""
        indexer = IncrementalIndexer()
        result = indexer.append_to_faiss([])
        
        # Should return 0
        assert result == 0
    
    def test_append_to_faiss_new_chunks(self):
        """Test appending new chunks to FAISS."""
        indexer = IncrementalIndexer()
        
        # Create test chunks
        test_chunks = [
            "This is a test chunk about machine learning",
            "Another test chunk about deep learning",
            "Third chunk about neural networks",
        ]
        
        initial_count = indexer.embeddings.shape[0] if indexer.embeddings is not None else 0
        embeddings_added = indexer.append_to_faiss(test_chunks)
        
        assert embeddings_added == len(test_chunks)
        assert indexer.embeddings is not None
        assert indexer.index is not None
        assert indexer.embeddings.shape[0] == initial_count + len(test_chunks)
    
    def test_process_incremental_batch_no_new_pdfs(self):
        """Test incremental batch with no new PDFs."""
        # Mark all PDFs as processed
        registry = load_registry()
        papers_dir = BASE_DIR / "data" / "papers"
        
        if papers_dir.exists():
            for pdf in papers_dir.glob("*.pdf"):
                if pdf.name not in registry:
                    registry[pdf.name] = {"processed": True}
        
        save_registry(registry)
        
        indexer = IncrementalIndexer()
        stats = indexer.process_incremental_batch()
        
        assert stats['new_pdfs_found'] == 0
        assert stats['pdfs_processed'] == []
        assert stats['index_updated'] == False
    
    def test_save_updated_index(self):
        """Test saving updated index."""
        indexer = IncrementalIndexer()
        
        # Create test data
        test_chunks = ["test chunk 1", "test chunk 2"]
        indexer.append_to_faiss(test_chunks)
        
        # Save
        indexer.save_updated_index()
        
        # Verify saved
        assert indexer.index is not None
        assert indexer.embeddings is not None
        assert indexer.chunks is not None


class TestVectorStoreHelpers:
    """Test vector store helper functions."""
    
    def test_append_embeddings_to_empty(self):
        """Test appending embeddings to empty array."""
        from src.vector_store import append_embeddings
        
        new_embeddings = np.random.randn(3, 384).astype("float32")
        result = append_embeddings(new_embeddings, None)
        
        assert result.shape == (3, 384)
        assert result.dtype == np.float32
    
    def test_append_embeddings_to_existing(self):
        """Test appending embeddings to existing array."""
        from src.vector_store import append_embeddings
        
        existing = np.random.randn(5, 384).astype("float32")
        new = np.random.randn(3, 384).astype("float32")
        
        result = append_embeddings(new, existing)
        
        assert result.shape == (8, 384)
        assert result.dtype == np.float32
    
    def test_get_index_stats_empty(self):
        """Test getting stats from empty/nonexistent index."""
        from src.vector_store import get_index_stats
        
        stats = get_index_stats()
        
        # Should return valid structure
        assert 'ntotal' in stats
        assert 'dimension' in stats
        assert 'chunk_count' in stats


class TestPaperRegistry:
    """Test paper registry updates."""
    
    def test_update_paper_new(self):
        """Test updating a new paper entry."""
        pdf_name = "test_new_paper.pdf"
        updates = {
            'processed': True,
            'pages': 10,
            'chunk_count': 50,
            'embedding_count': 50,
            'last_processed': '2026-06-23 12:00:00'
        }
        
        update_paper(pdf_name, updates)
        
        registry = load_registry()
        assert pdf_name in registry
        assert registry[pdf_name]['processed'] == True
        assert registry[pdf_name]['pages'] == 10
        assert registry[pdf_name]['chunk_count'] == 50
    
    def test_update_paper_existing(self):
        """Test updating an existing paper entry."""
        pdf_name = "test_existing_paper.pdf"
        
        # Initial update
        update_paper(pdf_name, {'processed': True, 'pages': 5})
        
        # Second update
        update_paper(pdf_name, {'chunk_count': 25})
        
        registry = load_registry()
        assert pdf_name in registry
        assert registry[pdf_name]['processed'] == True
        assert registry[pdf_name]['pages'] == 5
        assert registry[pdf_name]['chunk_count'] == 25


class TestIntegration:
    """Integration tests for incremental indexing."""
    
    def test_full_incremental_workflow(self):
        """Test complete incremental indexing workflow."""
        stats = run_incremental_indexing()
        
        # Should return valid stats
        assert isinstance(stats, dict)
        assert 'new_pdfs_found' in stats
        assert 'pdfs_processed' in stats
        assert 'pdfs_failed' in stats
        assert 'total_chunks_added' in stats
        assert 'total_embeddings_added' in stats
        assert 'total_pages' in stats
        assert 'index_updated' in stats
    
    def test_incremental_preserves_existing_data(self):
        """Test that incremental processing preserves existing data."""
        indexer = IncrementalIndexer()
        
        # Get initial state
        initial_index, initial_embeddings, initial_chunks = load_vector_store()
        
        if initial_embeddings is not None:
            initial_count = initial_embeddings.shape[0]
            
            # Add new chunks
            test_chunks = ["test chunk 1", "test chunk 2"]
            indexer.append_to_faiss(test_chunks)
            indexer.save_updated_index()
            
            # Verify data preserved
            updated_index, updated_embeddings, updated_chunks = load_vector_store()
            assert updated_embeddings.shape[0] == initial_count + len(test_chunks)


# Performance tests
class TestPerformance:
    """Performance tests for incremental indexing."""
    
    def test_chunk_processing_speed(self):
        """Test that chunk processing is reasonably fast."""
        import time
        
        indexer = IncrementalIndexer()
        papers_dir = BASE_DIR / "data" / "papers"
        
        if not papers_dir.exists():
            pytest.skip("Papers directory not found")
        
        pdfs = list(papers_dir.glob("*.pdf"))[:3]  # Test first 3
        if not pdfs:
            pytest.skip("No PDFs found")
        
        start_time = time.time()
        
        for pdf_path in pdfs:
            chunks, pages, chunk_count = indexer.process_new_pdf(pdf_path)
        
        elapsed = time.time() - start_time
        
        # Should process reasonably fast (< 60 seconds for 3 PDFs)
        assert elapsed < 60, f"Processing too slow: {elapsed:.2f}s"
    
    def test_embedding_append_speed(self):
        """Test that embedding appending is fast."""
        import time
        
        indexer = IncrementalIndexer()
        
        # Create test chunks
        test_chunks = [f"Test chunk {i}" for i in range(100)]
        
        start_time = time.time()
        indexer.append_to_faiss(test_chunks)
        elapsed = time.time() - start_time
        
        # Should append quickly (< 30 seconds for 100 chunks)
        assert elapsed < 30, f"Appending too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
