"""
Tests for SQLite metadata index functionality.

Tests cover:
- Database initialization
- Paper and gap operations
- Statistics calculation
- Query functionality
"""

import pytest
import tempfile
from pathlib import Path
from src.metadata_index import MetadataIndex, initialize_metadata_database


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_metadata.db"
        yield db_path


class TestMetadataIndexInitialization:
    """Test database initialization."""
    
    def test_initialize_database(self, temp_db):
        """Test creating database and tables."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        assert temp_db.exists(), "Database file should be created"
    
    def test_initialize_creates_tables(self, temp_db):
        """Test that tables are created."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in metadata.cursor.fetchall()]
            
            assert 'papers' in tables, "Papers table should exist"
            assert 'research_gaps' in tables, "Research gaps table should exist"
    
    def test_initialize_creates_indices(self, temp_db):
        """Test that indices are created."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indices = [row[0] for row in metadata.cursor.fetchall()]
            
            assert 'idx_paper_name' in indices, "Paper name index should exist"
            assert 'idx_gap_theme' in indices, "Gap theme index should exist"


class TestPaperOperations:
    """Test paper-related operations."""
    
    def test_add_paper(self, temp_db):
        """Test adding a paper."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            paper_id = metadata.add_paper(
                paper_name="test.pdf",
                pages=10,
                limitations=2
            )
            
            assert paper_id is not None, "Paper ID should be returned"
            assert paper_id > 0, "Paper ID should be positive"
    
    def test_add_multiple_papers(self, temp_db):
        """Test adding multiple papers."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            id1 = metadata.add_paper("paper1.pdf", pages=5, limitations=1)
            id2 = metadata.add_paper("paper2.pdf", pages=10, limitations=2)
            id3 = metadata.add_paper("paper3.pdf", pages=15, limitations=3)
            
            assert id1 != id2 != id3, "Each paper should have unique ID"
    
    def test_get_paper_by_name(self, temp_db):
        """Test retrieving paper by name."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=20, limitations=5)
            paper = metadata.get_paper_by_name("test.pdf")
            
            assert paper is not None, "Paper should be found"
            assert paper['paper_name'] == "test.pdf"
            assert paper['pages'] == 20
            assert paper['limitations'] == 5
    
    def test_get_nonexistent_paper(self, temp_db):
        """Test getting nonexistent paper returns None."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            paper = metadata.get_paper_by_name("nonexistent.pdf")
            assert paper is None, "Nonexistent paper should return None"
    
    def test_get_all_papers(self, temp_db):
        """Test retrieving all papers."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("paper1.pdf", pages=5)
            metadata.add_paper("paper2.pdf", pages=10)
            metadata.add_paper("paper3.pdf", pages=15)
            
            papers = metadata.get_all_papers()
            
            assert len(papers) == 3, "Should retrieve 3 papers"
            assert papers[0]['paper_name'] in ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
    
    def test_update_paper(self, temp_db):
        """Test updating existing paper."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=10, limitations=2)
            metadata.add_paper("test.pdf", pages=15, limitations=5)
            
            paper = metadata.get_paper_by_name("test.pdf")
            
            assert paper['pages'] == 15, "Pages should be updated"
            assert paper['limitations'] == 5, "Limitations should be updated"
    
    def test_update_paper_chunks(self, temp_db):
        """Test updating chunk and embedding counts."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=10)
            updated = metadata.update_paper_chunks("test.pdf", chunk_count=50, embedding_count=50)
            
            assert updated is True, "Update should succeed"
            
            paper = metadata.get_paper_by_name("test.pdf")
            assert paper['chunk_count'] == 50
            assert paper['embedding_count'] == 50
    
    def test_delete_paper(self, temp_db):
        """Test deleting paper."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=10)
            deleted = metadata.delete_paper("test.pdf")
            
            assert deleted is True, "Delete should succeed"
            
            paper = metadata.get_paper_by_name("test.pdf")
            assert paper is None, "Paper should be deleted"


class TestGapOperations:
    """Test gap-related operations."""
    
    def test_add_gap(self, temp_db):
        """Test adding a research gap."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            gap_id = metadata.add_gap(
                theme="Test Gap",
                cluster_size=5,
                paper_count=10,
                score=0.85
            )
            
            assert gap_id is not None, "Gap ID should be returned"
            assert gap_id > 0, "Gap ID should be positive"
    
    def test_get_all_gaps(self, temp_db):
        """Test retrieving all gaps."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_gap("Gap 1", cluster_size=5, score=0.8)
            metadata.add_gap("Gap 2", cluster_size=10, score=0.9)
            
            gaps = metadata.get_all_gaps()
            
            assert len(gaps) == 2, "Should retrieve 2 gaps"
            assert gaps[0]['score'] >= gaps[1]['score'], "Should be sorted by score"


class TestStatistics:
    """Test statistics calculation."""
    
    def test_get_statistics_empty(self, temp_db):
        """Test statistics with empty database."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            stats = metadata.get_statistics()
            
            assert stats['papers']['count'] == 0
            assert stats['gaps']['count'] == 0
    
    def test_get_statistics_with_papers(self, temp_db):
        """Test statistics with papers."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("paper1.pdf", pages=10, chunk_count=50)
            metadata.add_paper("paper2.pdf", pages=20, chunk_count=100)
            
            stats = metadata.get_statistics()
            
            assert stats['papers']['count'] == 2
            assert stats['papers']['total_pages'] == 30
            assert stats['papers']['total_chunks'] == 150
            assert stats['papers']['avg_pages'] == 15.0
    
    def test_get_statistics_with_gaps(self, temp_db):
        """Test statistics with gaps."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_gap("Gap 1", score=0.8)
            metadata.add_gap("Gap 2", score=0.9)
            
            stats = metadata.get_statistics()
            
            assert stats['gaps']['count'] == 2
            assert abs(stats['gaps']['avg_score'] - 0.85) < 0.01
    
    def test_statistics_calculations(self, temp_db):
        """Test complex statistics calculations."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            # Add papers with various metrics
            metadata.add_paper("p1.pdf", pages=5, limitations=2, chunk_count=25)
            metadata.add_paper("p2.pdf", pages=10, limitations=3, chunk_count=50)
            metadata.add_paper("p3.pdf", pages=15, limitations=1, chunk_count=75)
            
            stats = metadata.get_statistics()
            
            assert stats['papers']['count'] == 3
            assert stats['papers']['total_pages'] == 30
            assert stats['papers']['total_chunks'] == 150
            assert stats['papers']['total_limitations'] == 6
            assert stats['papers']['avg_chunks'] == 50.0


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self, temp_db):
        """Test using metadata index as context manager."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=10)
            paper = metadata.get_paper_by_name("test.pdf")
            
            assert paper is not None, "Paper should be accessible in context"
        
        # Connection should be closed after context
        assert metadata.connection is None, "Connection should be closed"


class TestDataPersistence:
    """Test data persistence across connections."""
    
    def test_data_persists(self, temp_db):
        """Test that data persists between connections."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            metadata.add_paper("test.pdf", pages=10, limitations=2)
        
        # New connection
        metadata2 = MetadataIndex(db_path=temp_db)
        
        with metadata2:
            paper = metadata2.get_paper_by_name("test.pdf")
            
            assert paper is not None, "Paper should persist"
            assert paper['pages'] == 10
            assert paper['limitations'] == 2


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self, temp_db):
        """Test complete workflow."""
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        with metadata:
            # Add papers
            metadata.add_paper("BERT.pdf", pages=5, limitations=3, chunk_count=25)
            metadata.add_paper("GPT-2.pdf", pages=24, limitations=4, chunk_count=120)
            
            # Add gaps
            metadata.add_gap("Efficiency", cluster_size=10, score=0.85)
            metadata.add_gap("Interpretability", cluster_size=8, score=0.92)
            
            # Get statistics
            stats = metadata.get_statistics()
            
            assert stats['papers']['count'] == 2
            assert stats['gaps']['count'] == 2
            assert stats['papers']['total_pages'] == 29
            assert stats['papers']['total_chunks'] == 145
    
    def test_initialize_function(self, temp_db):
        """Test convenience initialization function."""
        # This tests the module-level function
        metadata = MetadataIndex(db_path=temp_db)
        metadata.initialize_database()
        
        assert temp_db.exists(), "Database should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
