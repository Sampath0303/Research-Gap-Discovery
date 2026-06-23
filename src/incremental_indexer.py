"""
Incremental indexing for Research Gap Discovery Engine.

Processes only newly added PDFs, generates embeddings only for new chunks,
and appends to existing FAISS index without full rebuild.

Key features:
- Detect unprocessed PDFs from paper_registry.json
- Process only new PDFs (not already indexed)
- Generate embeddings only for new chunks
- Append embeddings to existing FAISS index
- Update registry to track processing status
- Preserve all existing data (no rebuilds)
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

from src.models import embedding_model
from src.chunker import RecursiveCharacterTextSplitter
from src.paper_registry import load_registry, save_registry, update_paper
from src.metadata_index import MetadataIndex


BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_FILE = (
    BASE_DIR
    / "outputs"
    / "faiss.index"
)

EMBEDDINGS_FILE = (
    BASE_DIR
    / "outputs"
    / "embeddings.npy"
)

CHUNKS_FILE = (
    BASE_DIR
    / "outputs"
    / "chunks.pkl"
)

PAPERS_DIR = BASE_DIR / "data" / "papers"


class IncrementalIndexer:
    """Handle incremental indexing without full rebuilds."""
    
    def __init__(self):
        """Initialize indexer with existing data."""
        self.registry = load_registry()
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        self.index = None
        self.embeddings = None
        self.chunks = None
        self.metadata_index = MetadataIndex()
        self.metadata_index.initialize_database()
        self.load_existing_data()
    
    def load_existing_data(self) -> None:
        """Load existing FAISS index, embeddings, and chunks."""
        if not all([
            FAISS_FILE.exists(),
            EMBEDDINGS_FILE.exists(),
            CHUNKS_FILE.exists()
        ]):
            return
        
        try:
            self.index = faiss.read_index(str(FAISS_FILE))
            self.embeddings = np.load(EMBEDDINGS_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load existing data: {e}")
    
    def detect_new_pdfs(self) -> List[Path]:
        """
        Detect PDFs that haven't been processed yet.
        
        Returns:
            List of Path objects for new PDFs
        """
        if not PAPERS_DIR.exists():
            return []
        
        new_pdfs = []
        
        for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
            filename = pdf_path.name
            
            # Check if in registry and marked as processed
            if filename not in self.registry or not self.registry[filename].get("processed"):
                new_pdfs.append(pdf_path)
        
        return new_pdfs
    
    def process_new_pdf(self, pdf_path: Path) -> Tuple[List[str], int, int]:
        """
        Extract text and create chunks from a new PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            (chunks_list, page_count, chunk_count)
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            page_count = len(pages)
            
            chunks = []
            for page in pages:
                text = page.page_content
                if text.strip():
                    page_chunks = self.chunker.split_text(text)
                    chunks.extend(page_chunks)
            
            return chunks, page_count, len(chunks)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path.name}: {e}")
            return [], 0, 0
    
    def append_to_faiss(self, new_chunks: List[str]) -> int:
        """
        Generate embeddings for new chunks and append to FAISS index.
        
        Args:
            new_chunks: List of new chunk texts
            
        Returns:
            Number of embeddings added
        """
        if not new_chunks:
            return 0
        
        # Generate embeddings for new chunks only
        new_embeddings = embedding_model.encode(
            new_chunks,
            show_progress_bar=False
        )
        new_embeddings = np.array(new_embeddings).astype("float32")
        
        # Append embeddings
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            # First time - create new index and embeddings
            self.embeddings = new_embeddings
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(new_embeddings)
        else:
            # Append to existing
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.index.add(new_embeddings)
        
        return len(new_chunks)
    
    def process_incremental_batch(self) -> Dict:
        """
        Process all new PDFs incrementally.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'new_pdfs_found': 0,
            'pdfs_processed': [],
            'pdfs_failed': [],
            'total_chunks_added': 0,
            'total_embeddings_added': 0,
            'total_pages': 0,
            'index_updated': False,
        }
        
        # Detect new PDFs
        new_pdfs = self.detect_new_pdfs()
        stats['new_pdfs_found'] = len(new_pdfs)
        
        if not new_pdfs:
            return stats
        
        print(f"\n[PROCESSING] Found {len(new_pdfs)} new PDF(s)")
        print("-" * 70)
        
        # Use metadata context manager for bulk operations
        with self.metadata_index:
            # Process each new PDF
            for pdf_path in new_pdfs:
                filename = pdf_path.name
                print(f"[PROCESS] {filename}...", end=" ")
                
                # Extract chunks
                chunks, pages, chunk_count = self.process_new_pdf(pdf_path)
                
                if not chunks:
                    print(f"[FAIL] No chunks extracted")
                    stats['pdfs_failed'].append(filename)
                    continue
                
                # Add to existing chunks list
                if self.chunks is None:
                    self.chunks = chunks
                else:
                    self.chunks.extend(chunks)
                
                # Append embeddings to FAISS
                embeddings_added = self.append_to_faiss(chunks)
                
                # Update registry
                update_paper(
                    filename,
                    {
                        'processed': True,
                        'pages': pages,
                        'chunk_count': chunk_count,
                        'embedding_count': embeddings_added,
                        'last_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                
                # Add to metadata database
                self.metadata_index.add_paper(
                    paper_name=filename,
                    pages=pages,
                    chunk_count=chunk_count,
                    embedding_count=embeddings_added
                )
                
                stats['pdfs_processed'].append(filename)
                stats['total_chunks_added'] += chunk_count
                stats['total_embeddings_added'] += embeddings_added
                stats['total_pages'] += pages
                
                print(f"OK ({pages} pages, {chunk_count} chunks)")
        
        # Save updated index and embeddings
        if stats['pdfs_processed']:
            self.save_updated_index()
            stats['index_updated'] = True
            print("-" * 70)
        
        return stats
    
    def save_updated_index(self) -> None:
        """Save updated FAISS index and embeddings without full rebuild."""
        if self.index is None or self.embeddings is None:
            print("[WARN] No index or embeddings to save")
            return
        
        # Create output directory
        FAISS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(FAISS_FILE))
        
        # Save embeddings
        np.save(EMBEDDINGS_FILE, self.embeddings)
        
        # Save chunks
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save updated registry
        save_registry(self.registry)
        
        print(f"[SAVED] Index: {FAISS_FILE}")
        print(f"[SAVED] Embeddings: {EMBEDDINGS_FILE}")
        print(f"[SAVED] Chunks: {CHUNKS_FILE}")


def add_new_chunks_to_index(new_chunks):
    """
    Add new chunks to existing FAISS index incrementally.
    
    Args:
        new_chunks: List of new LangChain Document chunks to add
        
    Returns:
        tuple: (updated_index, updated_embeddings, all_chunks)
    """
    
    if not new_chunks:
        raise ValueError("No new chunks provided.")
    
    # Load existing index and data
    if not all([
        FAISS_FILE.exists(),
        EMBEDDINGS_FILE.exists(),
        CHUNKS_FILE.exists()
    ]):
        raise ValueError(
            "Existing FAISS index not found. "
            "Create index first with create_vector_store()."
        )
    
    # Load existing data
    existing_index = faiss.read_index(str(FAISS_FILE))
    existing_embeddings = np.load(EMBEDDINGS_FILE)
    
    with open(CHUNKS_FILE, "rb") as file:
        existing_chunks = pickle.load(file)
    
    # Generate embeddings for new chunks
    new_texts = [chunk.page_content for chunk in new_chunks]
    
    new_embeddings = embedding_model.encode(
        new_texts,
        show_progress_bar=True
    )
    
    new_embeddings = np.array(new_embeddings).astype("float32")
    
    # Concatenate embeddings
    all_embeddings = np.vstack([
        existing_embeddings,
        new_embeddings
    ])
    
    # Add new embeddings to index
    existing_index.add(new_embeddings)
    
    # Combine all chunks
    all_chunks = existing_chunks + new_chunks
    
    # Save updated index and data
    save_incremental_update(
        existing_index,
        all_embeddings,
        all_chunks
    )
    
    return (
        existing_index,
        all_embeddings,
        all_chunks
    )


def save_incremental_update(
    index,
    embeddings,
    chunks
):
    """
    Save updated FAISS index and embeddings.
    """
    
    FAISS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    faiss.write_index(
        index,
        str(FAISS_FILE)
    )
    
    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )
    
    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:
        pickle.dump(
            chunks,
            file
        )
    
    print(
        f"FAISS index updated -> {FAISS_FILE}"
    )
    print(
        f"Total chunks: {len(chunks)}"
    )


def run_incremental_indexing() -> Dict:
    """
    Run incremental indexing on all new PDFs.
    
    Returns:
        Statistics dictionary
    """
    indexer = IncrementalIndexer()
    stats = indexer.process_incremental_batch()
    
    return stats


if __name__ == "__main__":
    stats = run_incremental_indexing()
    
    print("\n" + "="*70)
    print("INCREMENTAL INDEXING SUMMARY")
    print("="*70)
    print(f"New PDFs found:        {stats['new_pdfs_found']}")
    print(f"PDFs processed:        {len(stats['pdfs_processed'])}")
    print(f"PDFs failed:           {len(stats['pdfs_failed'])}")
    print(f"Total chunks added:    {stats['total_chunks_added']}")
    print(f"Total embeddings:      {stats['total_embeddings_added']}")
    print(f"Total pages:           {stats['total_pages']}")
    print(f"Index updated:         {stats['index_updated']}")
    print("="*70)
    
    if stats['pdfs_processed']:
        print("\nProcessed PDFs:")
        for pdf in stats['pdfs_processed']:
            print(f"  [OK] {pdf}")
    
    if stats['pdfs_failed']:
        print("\nFailed PDFs:")
        for pdf in stats['pdfs_failed']:
            print(f"  [FAIL] {pdf}")
