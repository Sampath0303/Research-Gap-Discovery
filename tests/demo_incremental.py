"""
Demo script showing incremental indexing functionality.

Demonstrates:
1. Detecting new PDFs
2. Processing incrementally
3. Updating registry
4. Appending to FAISS
"""

import warnings
warnings.filterwarnings('ignore')

from src.incremental_indexer import run_incremental_indexing
from src.vector_store import get_index_stats
from src.paper_registry import load_registry

print("\n" + "="*80)
print("INCREMENTAL INDEXING DEMONSTRATION")
print("="*80 + "\n")

# Show initial index stats
print("[1] INITIAL INDEX STATUS")
print("-" * 80)
stats_before = get_index_stats()
print(f"  Index vectors: {stats_before['ntotal']}")
print(f"  Dimension:     {stats_before['dimension']}")
print(f"  Chunk count:   {stats_before['chunk_count']}")

# Show registry
print("\n[2] PAPER REGISTRY")
print("-" * 80)
registry = load_registry()
print(f"  Registered papers: {len(registry)}")
processed = sum(1 for p in registry.values() if p.get('processed'))
unprocessed = len(registry) - processed
print(f"  Processed:        {processed}")
print(f"  Unprocessed:      {unprocessed}")

# Run incremental indexing
print("\n[3] RUNNING INCREMENTAL INDEXING")
print("-" * 80)
result = run_incremental_indexing()

# Display results
print("\n[4] INCREMENTAL INDEXING RESULTS")
print("-" * 80)
print(f"  New PDFs found:       {result['new_pdfs_found']}")
print(f"  PDFs processed:       {len(result['pdfs_processed'])}")
print(f"  PDFs failed:          {len(result['pdfs_failed'])}")
print(f"  Chunks added:         {result['total_chunks_added']}")
print(f"  Embeddings added:     {result['total_embeddings_added']}")
print(f"  Pages processed:      {result['total_pages']}")
print(f"  Index updated:        {result['index_updated']}")

if result['pdfs_processed']:
    print("\n  Processed files:")
    for pdf in result['pdfs_processed'][:5]:
        print(f"    - {pdf}")
    if len(result['pdfs_processed']) > 5:
        print(f"    ... and {len(result['pdfs_processed']) - 5} more")

# Show final index stats
print("\n[5] FINAL INDEX STATUS")
print("-" * 80)
stats_after = get_index_stats()
print(f"  Index vectors: {stats_after['ntotal']}")
print(f"  Dimension:     {stats_after['dimension']}")
print(f"  Chunk count:   {stats_after['chunk_count']}")

# Calculate changes
vector_increase = stats_after['ntotal'] - stats_before['ntotal']
chunk_increase = stats_after['chunk_count'] - stats_before['chunk_count']

print(f"\n  Change in vectors: +{vector_increase}")
print(f"  Change in chunks:  +{chunk_increase}")

print("\n" + "="*80)
print("DEMONSTRATION COMPLETE")
print("="*80 + "\n")
