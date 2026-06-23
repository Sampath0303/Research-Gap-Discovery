"""
SQLite Metadata Layer for Research Gap Discovery Engine.

Manages metadata storage for papers and research gaps using SQLite.
Enables efficient querying and statistics generation.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "outputs" / "metadata.db"


class MetadataIndex:
    """SQLite metadata index for papers and research gaps."""
    
    def __init__(self, db_path: Path = None):
        """Initialize metadata index with database connection."""
        self.db_path = db_path or DB_FILE
        self.connection = None
        self.cursor = None
    
    def initialize_database(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_name TEXT UNIQUE NOT NULL,
                pages INTEGER,
                limitations INTEGER DEFAULT 0,
                processed_date TEXT,
                chunk_count INTEGER DEFAULT 0,
                embedding_count INTEGER DEFAULT 0
            )
        """)
        
        # Create research_gaps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT NOT NULL,
                cluster_size INTEGER DEFAULT 0,
                paper_count INTEGER DEFAULT 0,
                score REAL DEFAULT 0.0,
                created_date TEXT
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paper_name 
            ON papers(paper_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_gap_theme 
            ON research_gaps(theme)
        """)
        
        conn.commit()
        conn.close()
        
        print(f"[OK] Database initialized: {self.db_path}")
    
    def connect(self) -> None:
        """Establish database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.commit()
            self.connection.close()
            self.connection = None
            self.cursor = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def add_paper(
        self,
        paper_name: str,
        pages: int,
        limitations: int = 0,
        chunk_count: int = 0,
        embedding_count: int = 0
    ) -> int:
        """
        Add or update paper metadata.
        
        Args:
            paper_name: Name of the PDF file
            pages: Number of pages
            limitations: Number of limitations found
            chunk_count: Number of chunks generated
            embedding_count: Number of embeddings created
            
        Returns:
            Paper ID
        """
        self.connect()
        
        processed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            self.cursor.execute("""
                INSERT INTO papers 
                (paper_name, pages, limitations, chunk_count, embedding_count, processed_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (paper_name, pages, limitations, chunk_count, embedding_count, processed_date))
            
            self.connection.commit()
            paper_id = self.cursor.lastrowid
            
            return paper_id
            
        except sqlite3.IntegrityError:
            # Paper already exists, update it
            self.cursor.execute("""
                UPDATE papers 
                SET pages = ?, limitations = ?, chunk_count = ?, 
                    embedding_count = ?, processed_date = ?
                WHERE paper_name = ?
            """, (pages, limitations, chunk_count, embedding_count, processed_date, paper_name))
            
            self.connection.commit()
            
            # Get paper ID
            self.cursor.execute("SELECT id FROM papers WHERE paper_name = ?", (paper_name,))
            result = self.cursor.fetchone()
            return result[0] if result else None
    
    def add_gap(
        self,
        theme: str,
        cluster_size: int = 0,
        paper_count: int = 0,
        score: float = 0.0
    ) -> int:
        """
        Add research gap.
        
        Args:
            theme: Gap theme/title
            cluster_size: Size of gap cluster
            paper_count: Number of papers related to gap
            score: Gap importance score
            
        Returns:
            Gap ID
        """
        self.connect()
        
        created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.cursor.execute("""
            INSERT INTO research_gaps 
            (theme, cluster_size, paper_count, score, created_date)
            VALUES (?, ?, ?, ?, ?)
        """, (theme, cluster_size, paper_count, score, created_date))
        
        self.connection.commit()
        return self.cursor.lastrowid
    
    def get_all_papers(self) -> List[Dict]:
        """
        Get all papers from database.
        
        Returns:
            List of paper dictionaries
        """
        self.connect()
        
        self.cursor.execute("""
            SELECT id, paper_name, pages, limitations, chunk_count, 
                   embedding_count, processed_date
            FROM papers
            ORDER BY processed_date DESC
        """)
        
        rows = self.cursor.fetchall()
        
        papers = []
        for row in rows:
            papers.append({
                'id': row[0],
                'paper_name': row[1],
                'pages': row[2],
                'limitations': row[3],
                'chunk_count': row[4],
                'embedding_count': row[5],
                'processed_date': row[6],
            })
        
        return papers
    
    def get_all_gaps(self) -> List[Dict]:
        """
        Get all research gaps from database.
        
        Returns:
            List of gap dictionaries
        """
        self.connect()
        
        self.cursor.execute("""
            SELECT id, theme, cluster_size, paper_count, score, created_date
            FROM research_gaps
            ORDER BY score DESC
        """)
        
        rows = self.cursor.fetchall()
        
        gaps = []
        for row in rows:
            gaps.append({
                'id': row[0],
                'theme': row[1],
                'cluster_size': row[2],
                'paper_count': row[3],
                'score': row[4],
                'created_date': row[5],
            })
        
        return gaps
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        self.connect()
        
        # Get paper statistics
        self.cursor.execute("SELECT COUNT(*) FROM papers")
        paper_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT SUM(pages) FROM papers")
        total_pages = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT SUM(chunk_count) FROM papers")
        total_chunks = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT SUM(embedding_count) FROM papers")
        total_embeddings = self.cursor.fetchone()[0] or 0
        
        self.cursor.execute("SELECT SUM(limitations) FROM papers")
        total_limitations = self.cursor.fetchone()[0] or 0
        
        # Get gap statistics
        self.cursor.execute("SELECT COUNT(*) FROM research_gaps")
        gap_count = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT AVG(score) FROM research_gaps")
        avg_gap_score = self.cursor.fetchone()[0] or 0.0
        
        self.cursor.execute("SELECT SUM(paper_count) FROM research_gaps")
        gap_papers_count = self.cursor.fetchone()[0] or 0
        
        return {
            'papers': {
                'count': paper_count,
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_embeddings': total_embeddings,
                'total_limitations': total_limitations,
                'avg_pages': total_pages / paper_count if paper_count > 0 else 0,
                'avg_chunks': total_chunks / paper_count if paper_count > 0 else 0,
            },
            'gaps': {
                'count': gap_count,
                'avg_score': avg_gap_score,
                'total_papers_in_gaps': gap_papers_count,
            },
        }
    
    def get_paper_by_name(self, paper_name: str) -> Optional[Dict]:
        """
        Get specific paper by name.
        
        Args:
            paper_name: Name of the paper
            
        Returns:
            Paper dictionary or None
        """
        self.connect()
        
        self.cursor.execute("""
            SELECT id, paper_name, pages, limitations, chunk_count, 
                   embedding_count, processed_date
            FROM papers
            WHERE paper_name = ?
        """, (paper_name,))
        
        row = self.cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'paper_name': row[1],
                'pages': row[2],
                'limitations': row[3],
                'chunk_count': row[4],
                'embedding_count': row[5],
                'processed_date': row[6],
            }
        
        return None
    
    def delete_paper(self, paper_name: str) -> bool:
        """
        Delete paper from database.
        
        Args:
            paper_name: Name of the paper to delete
            
        Returns:
            True if deleted, False otherwise
        """
        self.connect()
        
        self.cursor.execute("DELETE FROM papers WHERE paper_name = ?", (paper_name,))
        self.connection.commit()
        
        return self.cursor.rowcount > 0
    
    def update_paper_chunks(
        self,
        paper_name: str,
        chunk_count: int,
        embedding_count: int
    ) -> bool:
        """
        Update chunk and embedding counts for a paper.
        
        Args:
            paper_name: Name of the paper
            chunk_count: Number of chunks
            embedding_count: Number of embeddings
            
        Returns:
            True if updated, False otherwise
        """
        self.connect()
        
        self.cursor.execute("""
            UPDATE papers 
            SET chunk_count = ?, embedding_count = ?
            WHERE paper_name = ?
        """, (chunk_count, embedding_count, paper_name))
        
        self.connection.commit()
        
        return self.cursor.rowcount > 0
    
    def export_to_csv(self, output_file: Path) -> None:
        """
        Export papers and gaps to CSV files.
        
        Args:
            output_file: Path to export directory
        """
        import csv
        
        output_file.mkdir(parents=True, exist_ok=True)
        
        # Export papers
        papers = self.get_all_papers()
        if papers:
            papers_csv = output_file / "papers.csv"
            with open(papers_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=papers[0].keys())
                writer.writeheader()
                writer.writerows(papers)
            print(f"[OK] Exported papers to {papers_csv}")
        
        # Export gaps
        gaps = self.get_all_gaps()
        if gaps:
            gaps_csv = output_file / "gaps.csv"
            with open(gaps_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=gaps[0].keys())
                writer.writeheader()
                writer.writerows(gaps)
            print(f"[OK] Exported gaps to {gaps_csv}")


def initialize_metadata_database() -> MetadataIndex:
    """
    Initialize and return metadata index.
    
    Returns:
        MetadataIndex instance
    """
    metadata = MetadataIndex()
    metadata.initialize_database()
    return metadata


if __name__ == "__main__":
    # Demo usage
    metadata = initialize_metadata_database()
    
    with metadata:
        # Add sample papers
        metadata.add_paper("BERT.pdf", pages=5, limitations=3, chunk_count=25)
        metadata.add_paper("GPT-2.pdf", pages=24, limitations=4, chunk_count=120)
        metadata.add_paper("Transformers.pdf", pages=15, limitations=2, chunk_count=75)
        
        # Add sample gaps
        metadata.add_gap("Efficiency of Large Models", cluster_size=10, paper_count=3, score=0.85)
        metadata.add_gap("Interpretability", cluster_size=8, paper_count=3, score=0.92)
        
        # Get statistics
        stats = metadata.get_statistics()
        
        print("\n" + "="*70)
        print("METADATA DATABASE STATISTICS")
        print("="*70)
        print(f"\nPapers:")
        print(f"  Count: {stats['papers']['count']}")
        print(f"  Total pages: {stats['papers']['total_pages']}")
        print(f"  Total chunks: {stats['papers']['total_chunks']}")
        print(f"  Avg pages/paper: {stats['papers']['avg_pages']:.1f}")
        
        print(f"\nResearch Gaps:")
        print(f"  Count: {stats['gaps']['count']}")
        print(f"  Avg score: {stats['gaps']['avg_score']:.2f}")
        
        print(f"\nAll Papers:")
        for paper in metadata.get_all_papers():
            print(f"  - {paper['paper_name']}: {paper['pages']} pages, {paper['chunk_count']} chunks")
        
        print("="*70 + "\n")
