"""
Quick performance test - 50 PDFs baseline.

Tests V1.5 performance on smaller dataset.
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import warnings

warnings.filterwarnings('ignore')

from tests import _bootstrap
from _bootstrap import BASE_DIR
from src.chunker import RecursiveCharacterTextSplitter


class PerformanceProfiler:
    """Profile performance metrics for different dataset sizes."""
    
    def __init__(self, dataset_dir: Path):
        """Initialize profiler with dataset directory."""
        self.dataset_dir = dataset_dir
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from dataset directory."""
        return sorted(self.dataset_dir.glob("*.pdf"))
    
    def load_and_chunk_pdfs(self, pdf_files: List[Path]) -> Tuple[int, int, float, List[str]]:
        """
        Load PDFs and create chunks.
        
        Returns:
            (page_count, chunk_count, elapsed_time_seconds, chunks)
        """
        start_time = time.time()
        
        all_chunks = []
        total_pages = 0
        
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                total_pages += len(pages)
                
                for page in pages:
                    text = page.page_content
                    if text.strip():
                        chunks = self.chunker.split_text(text)
                        all_chunks.extend(chunks)
            except Exception as e:
                pass
                continue
        
        elapsed = time.time() - start_time
        return total_pages, len(all_chunks), elapsed, all_chunks
    
    def generate_embeddings(self, chunks: List[str]) -> Tuple[np.ndarray, float]:
        """Generate embeddings for chunks."""
        if not chunks:
            return np.array([]), 0.0
        
        start_time = time.time()
        
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        elapsed = time.time() - start_time
        
        return embeddings, elapsed
    
    def build_index(self, embeddings: np.ndarray) -> Tuple[object, float]:
        """Build FAISS index."""
        if embeddings.shape[0] == 0:
            return None, 0.0
        
        start_time = time.time()
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        elapsed = time.time() - start_time
        
        return index, elapsed
    
    def measure_search_latency(self, chunks: List[str], index: object, k: int = 5) -> Tuple[float, float, float]:
        """Measure search latency (p50, p95, p99)."""
        if not chunks or index is None:
            return 0.0, 0.0, 0.0
        
        num_queries = min(10, len(chunks))
        query_indices = np.random.choice(len(chunks), num_queries, replace=False)
        queries = [chunks[i] for i in query_indices]
        
        latencies = []
        
        for query in queries:
            query_embedding = self.embedding_model.encode([query])[0]
            
            start_time = time.perf_counter()
            distances, indices = index.search(
                np.array([query_embedding], dtype='float32'),
                k=k
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            latencies.append(elapsed_ms)
        
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        return p50, p95, p99
    
    def profile_dataset(self) -> Dict:
        """Profile complete dataset."""
        pdf_files = self.get_pdf_files()
        pdf_count = len(pdf_files)
        
        if pdf_count == 0:
            return None
        
        # Load and chunk
        page_count, chunk_count, chunk_time, all_chunks = self.load_and_chunk_pdfs(pdf_files)
        
        # Generate embeddings
        embeddings, embed_time = self.generate_embeddings(all_chunks)
        
        # Build FAISS index
        index, build_time = self.build_index(embeddings)
        
        # Measure search latency
        p50, p95, p99 = self.measure_search_latency(all_chunks, index)
        
        return {
            "pdf_count": pdf_count,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "chunk_time_sec": chunk_time,
            "embed_time_sec": embed_time,
            "build_time_sec": build_time,
            "search_p50_ms": p50,
            "search_p95_ms": p95,
            "search_p99_ms": p99,
        }


def run_performance_tests() -> None:
    """Run performance tests."""
    base_dir = Path(BASE_DIR)
    
    print("\n" + "="*90)
    print("PERFORMANCE TESTING - V1.5 BASELINE METRICS")
    print("="*90 + "\n")
    
    dataset_dir = base_dir / "data" / "scaled_papers_50"
    
    if not dataset_dir.exists():
        print("[ERROR] Dataset directory not found:", dataset_dir)
        return
    
    pdf_count = len(list(dataset_dir.glob("*.pdf")))
    print(f"[TESTING] 50 PDF Dataset ({pdf_count} PDFs)\n")
    
    profiler = PerformanceProfiler(dataset_dir)
    metrics = profiler.profile_dataset()
    
    if not metrics:
        print("[ERROR] Failed to profile dataset")
        return
    
    # Display results
    print("\n" + "="*90)
    print("PERFORMANCE RESULTS TABLE")
    print("="*90 + "\n")
    
    print(f"{'Metric':<30} {'Value':<20} {'Unit'}")
    print("-" * 90)
    print(f"{'PDF Count':<30} {metrics['pdf_count']:<20} files")
    print(f"{'Page Count':<30} {metrics['page_count']:<20} pages")
    print(f"{'Chunk Count':<30} {metrics['chunk_count']:<20} chunks")
    print(f"{'Chunk Generation Time':<30} {metrics['chunk_time_sec']:<20.2f} seconds")
    print(f"{'Embedding Generation Time':<30} {metrics['embed_time_sec']:<20.2f} seconds")
    print(f"{'FAISS Build Time':<30} {metrics['build_time_sec']:<20.2f} seconds")
    print(f"{'Total Build Time':<30} {(metrics['chunk_time_sec'] + metrics['embed_time_sec'] + metrics['build_time_sec']):<20.2f} seconds")
    print("-" * 90)
    print(f"{'Search Latency (P50)':<30} {metrics['search_p50_ms']:<20.2f} ms")
    print(f"{'Search Latency (P95)':<30} {metrics['search_p95_ms']:<20.2f} ms")
    print(f"{'Search Latency (P99)':<30} {metrics['search_p99_ms']:<20.2f} ms")
    print("=" * 90)
    
    # Projections for larger datasets
    print("\n" + "="*90)
    print("PERFORMANCE PROJECTIONS (Estimated based on 50 PDFs)")
    print("="*90 + "\n")
    
    print(f"{'Dataset Size':<15} {'PDFs':<8} {'Chunks':<10} {'Build Time':<15} {'Search (P50)':<15}")
    print("-" * 90)
    
    base_pdfs = metrics['pdf_count']
    base_chunks = metrics['chunk_count']
    base_build_time = metrics['chunk_time_sec'] + metrics['embed_time_sec'] + metrics['build_time_sec']
    base_search = metrics['search_p50_ms']
    
    for factor, label in [(1, "50 PDFs"), (2, "100 PDFs"), (5, "250 PDFs"), (10, "500 PDFs")]:
        est_pdfs = base_pdfs * factor
        est_chunks = base_chunks * factor
        est_build = base_build_time * factor
        est_search = base_search * (factor ** 0.7)  # Sublinear scaling
        
        print(f"{label:<15} {est_pdfs:<8} {est_chunks:<10} {est_build:<15.2f}s {est_search:<15.2f}ms")
    
    print("-" * 90)
    print("\nNote: Estimates assume linear chunk growth and sublinear search scaling (O(n^0.7))")
    print("="*90 + "\n")


if __name__ == "__main__":
    run_performance_tests()
