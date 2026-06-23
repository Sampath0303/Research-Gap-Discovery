"""
Performance testing and benchmarking for V1.5 implementation.

Measures:
- PDF count and page count
- Chunk generation time and count
- Embedding generation time
- FAISS index build time
- Search latency (p50, p95, p99)

Uses existing V1.5 pipeline without any modifications.
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss

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
    
    def load_and_chunk_pdfs(self, pdf_files: List[Path]) -> Tuple[int, int, float]:
        """
        Load PDFs and create chunks.
        
        Returns:
            (page_count, chunk_count, elapsed_time_seconds)
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
                print(f"    [WARN] Error loading {pdf_path.name}: {e}")
                continue
        
        elapsed = time.time() - start_time
        return total_pages, len(all_chunks), elapsed
    
    def generate_embeddings(self, chunks: List[str]) -> Tuple[np.ndarray, float]:
        """
        Generate embeddings for chunks.
        
        Returns:
            (embeddings_array, elapsed_time_seconds)
        """
        if not chunks:
            return np.array([]), 0.0
        
        start_time = time.time()
        
        # Batch encode chunks
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
        """
        Build FAISS index.
        
        Returns:
            (faiss_index, elapsed_time_seconds)
        """
        if embeddings.shape[0] == 0:
            return None, 0.0
        
        start_time = time.time()
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        elapsed = time.time() - start_time
        return index, elapsed
    
    def measure_search_latency(self, chunks: List[str], index: object, k: int = 5) -> Tuple[float, float, float]:
        """
        Measure search latency (p50, p95, p99).
        
        Uses sample queries.
        
        Returns:
            (p50_ms, p95_ms, p99_ms)
        """
        if not chunks or index is None:
            return 0.0, 0.0, 0.0
        
        # Sample queries from chunks
        num_queries = min(10, len(chunks))
        query_indices = np.random.choice(len(chunks), num_queries, replace=False)
        queries = [chunks[i] for i in query_indices]
        
        latencies = []
        
        for query in queries:
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search
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
        """Profile complete dataset and return metrics."""
        pdf_files = self.get_pdf_files()
        pdf_count = len(pdf_files)
        
        if pdf_count == 0:
            return None
        
        # Step 1: Load and chunk PDFs
        page_count, chunk_count, chunk_time = self.load_and_chunk_pdfs(pdf_files)
        
        # Reload for embeddings (to get chunk objects)
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                for page in pages:
                    text = page.page_content
                    if text.strip():
                        chunks = self.chunker.split_text(text)
                        all_chunks.extend(chunks)
            except:
                continue
        
        # Step 2: Generate embeddings
        embeddings, embed_time = self.generate_embeddings(all_chunks)
        
        # Step 3: Build FAISS index
        index, build_time = self.build_index(embeddings)
        
        # Step 4: Measure search latency
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
    """Run performance tests on all dataset sizes."""
    base_dir = Path(BASE_DIR)
    
    dataset_configs = [
        (50, "scaled_papers_50"),
        (100, "scaled_papers_100"),
        (250, "scaled_papers_250"),
        (500, "scaled_papers_500"),
    ]
    
    print("\n" + "="*80)
    print("PERFORMANCE TESTING - V1.5 BASELINE")
    print("="*80 + "\n")
    
    results = []
    
    for num_pdfs, folder_name in dataset_configs:
        dataset_dir = base_dir / "data" / folder_name
        
        if not dataset_dir.exists():
            print(f"[SKIP] {folder_name}: Directory not found")
            continue
        
        pdf_count = len(list(dataset_dir.glob("*.pdf")))
        if pdf_count == 0:
            print(f"[SKIP] {folder_name}: No PDFs found")
            continue
        
        print(f"[TEST] {folder_name} ({pdf_count} PDFs)...")
        
        profiler = PerformanceProfiler(dataset_dir)
        metrics = profiler.profile_dataset()
        
        if metrics:
            results.append(metrics)
            print(f"       - Pages: {metrics['page_count']}, Chunks: {metrics['chunk_count']}")
            print(f"       - Chunk Time: {metrics['chunk_time_sec']:.2f}s")
            print(f"       - Embed Time: {metrics['embed_time_sec']:.2f}s")
            print(f"       - Build Time: {metrics['build_time_sec']:.2f}s")
            print(f"       - Search P50: {metrics['search_p50_ms']:.2f}ms")
            print()
    
    # Print results table
    print("\n" + "="*80)
    print("PERFORMANCE REPORT TABLE")
    print("="*80 + "\n")
    
    if results:
        print(f"{'PDFs':<6} {'Pages':<6} {'Chunks':<8} {'Chunk[s]':<10} {'Embed[s]':<10} {'Build[s]':<10} {'P50[ms]':<10} {'P95[ms]':<10} {'P99[ms]':<10}")
        print("-" * 88)
        
        for r in results:
            print(
                f"{r['pdf_count']:<6} "
                f"{r['page_count']:<6} "
                f"{r['chunk_count']:<8} "
                f"{r['chunk_time_sec']:<10.2f} "
                f"{r['embed_time_sec']:<10.2f} "
                f"{r['build_time_sec']:<10.2f} "
                f"{r['search_p50_ms']:<10.2f} "
                f"{r['search_p95_ms']:<10.2f} "
                f"{r['search_p99_ms']:<10.2f}"
            )
        
        print("-" * 88)
        
        # Calculate growth rates
        if len(results) > 1:
            print("\nGROWTH ANALYSIS:")
            print("-" * 88)
            
            for i in range(1, len(results)):
                prev = results[i-1]
                curr = results[i]
                
                pdf_growth = curr["pdf_count"] / prev["pdf_count"]
                chunk_growth = curr["chunk_count"] / prev["chunk_count"]
                embed_growth = curr["embed_time_sec"] / prev["embed_time_sec"] if prev["embed_time_sec"] > 0 else 0
                search_growth = curr["search_p50_ms"] / prev["search_p50_ms"] if prev["search_p50_ms"] > 0 else 0
                
                print(
                    f"  {prev['pdf_count']} -> {curr['pdf_count']} PDFs: "
                    f"Chunks grow {chunk_growth:.2f}x, "
                    f"Embed time {embed_growth:.2f}x, "
                    f"Search latency {search_growth:.2f}x"
                )
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    run_performance_tests()
