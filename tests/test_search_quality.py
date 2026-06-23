"""
Retrieval Quality Evaluation Test.
Evaluates the quality of semantic search without using Gemini API.
"""

from tests import _bootstrap
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from src.models import embedding_model
from src.vector_store import load_vector_store
from src.hybrid_retriever import search_hybrid
from src.paper_registry import load_registry


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


# Sample queries designed to evaluate retrieval quality
SAMPLE_QUERIES = [
    "attention mechanisms in transformer models",
    "BERT pre-training and fine-tuning strategies",
    "handling long-range dependencies in neural networks",
    "data augmentation techniques for NLP",
    "multilingual model training approaches",
    "distillation methods for model compression",
    "optimization algorithms for deep learning",
    "interpretability and explainability in transformers",
    "positional encoding in sequence models",
    "scaling language models to production",
]


def load_paper_metadata() -> Dict[str, Dict]:
    """Load paper metadata from registry."""
    registry = load_registry()
    return registry


def calculate_embedding_similarity(
    query_embedding: np.ndarray,
    doc_embedding: np.ndarray
) -> float:
    """
    Calculate cosine similarity between query and document embeddings.
    Returns value between 0 and 1 (1 = identical, 0 = orthogonal).
    """
    # Normalize vectors
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    doc_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-8)
    
    # Cosine similarity
    similarity = float(np.dot(query_norm, doc_norm))
    
    # Convert to 0-1 range (cosine similarity is -1 to 1)
    return (similarity + 1) / 2


def convert_l2_distance_to_similarity(
    distance: float, 
    embeddings: np.ndarray
) -> float:
    """
    Convert L2 distance to similarity score (0-1).
    Uses the max distance in embeddings to normalize.
    """
    # Calculate max possible L2 distance for normalized vectors
    # For normalized vectors, L2 distance ranges from 0 to 2
    max_distance = 2.0
    
    # Convert to similarity: higher similarity for lower distance
    similarity = max(0, 1 - (distance / max_distance))
    return min(1.0, max(0, similarity))


def evaluate_query(
    query: str,
    index,
    embeddings: np.ndarray,
    chunks: List,
    top_k: int = 5
) -> Dict:
    """
    Evaluate retrieval quality for a single query.
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Get query embedding
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Normalize query embedding
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    # Retrieve results (these are already scored by FAISS)
    retrieved_chunks = search_hybrid(query, index, embeddings, chunks, k=top_k)
    
    # Extract papers and calculate direct cosine similarities
    papers_retrieved = []
    total_relevance_score = 0.0
    unique_papers = set()
    
    for rank, result in enumerate(retrieved_chunks, 1):
        paper_name = result.get("paper", "Unknown")
        content = result.get("content", "")
        
        # Generate embedding for the retrieved content to compute cosine similarity
        try:
            chunk_embedding = embedding_model.encode([content])
            chunk_embedding = np.array(chunk_embedding).astype("float32")
            chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
            
            # Calculate cosine similarity (0 to 1)
            cosine_sim = float(np.dot(query_norm[0], chunk_norm[0]))
            # Convert from [-1, 1] to [0, 1]
            similarity = (cosine_sim + 1) / 2
        except:
            # Fallback: use normalized distance if embedding fails
            distance = result.get("score", 2.0)
            similarity = max(0, 1 - (distance / 2.0))
        
        unique_papers.add(paper_name)
        total_relevance_score += similarity
        
        papers_retrieved.append({
            "rank": rank,
            "paper": paper_name,
            "page": result.get("page", "?"),
            "distance": result.get("score", 0.0),
            "similarity": similarity,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        })
    
    # Calculate metrics
    avg_similarity = (
        total_relevance_score / len(papers_retrieved)
        if papers_retrieved
        else 0.0
    )
    
    num_unique_papers = len(unique_papers)
    
    # Quality score (0-100)
    # Based on: average similarity (70%), paper diversity (30%)
    diversity_score = (num_unique_papers / top_k) * 100  # 0-100
    quality_score = (avg_similarity * 100 * 0.7) + (diversity_score * 0.3)
    
    return {
        "query": query,
        "retrieved_count": len(papers_retrieved),
        "unique_papers": num_unique_papers,
        "papers": papers_retrieved,
        "avg_similarity": avg_similarity,
        "diversity_score": diversity_score,
        "quality_score": quality_score,
        "timestamp": datetime.now().isoformat()
    }


def generate_evaluation_report(results: List[Dict]) -> str:
    """Generate a formatted evaluation report."""
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("RETRIEVAL QUALITY EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    total_queries = len(results)
    avg_quality = np.mean([r["quality_score"] for r in results])
    avg_similarity = np.mean([r["avg_similarity"] for r in results])
    avg_diversity = np.mean([r["diversity_score"] for r in results])
    
    report_lines.append("OVERALL METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Queries Evaluated: {total_queries}")
    report_lines.append(f"Average Quality Score: {avg_quality:.2f}/100")
    report_lines.append(f"Average Similarity: {avg_similarity:.4f}")
    report_lines.append(f"Average Diversity: {avg_diversity:.2f}/100")
    report_lines.append("")
    
    # Quality interpretation
    if avg_quality >= 80:
        quality_level = "EXCELLENT"
    elif avg_quality >= 70:
        quality_level = "GOOD"
    elif avg_quality >= 60:
        quality_level = "FAIR"
    elif avg_quality >= 50:
        quality_level = "POOR"
    else:
        quality_level = "CRITICAL"
    
    report_lines.append(f"Quality Level: {quality_level}")
    report_lines.append("")
    
    # Detailed results per query
    report_lines.append("DETAILED RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for idx, result in enumerate(results, 1):
        report_lines.append(f"Query #{idx}: {result['query']}")
        report_lines.append("-" * 80)
        report_lines.append(f"  Quality Score: {result['quality_score']:.2f}/100")
        report_lines.append(f"  Avg Similarity: {result['avg_similarity']:.4f}")
        report_lines.append(f"  Diversity: {result['diversity_score']:.2f}%")
        report_lines.append(f"  Retrieved Papers: {result['unique_papers']} unique")
        report_lines.append("")
        
        report_lines.append("  Top Retrieved Papers:")
        for paper in result["papers"]:
            report_lines.append(
                f"    Rank {paper['rank']}: {paper['paper']} "
                f"(Page {paper['page']}, Similarity: {paper['similarity']:.4f})"
            )
        
        report_lines.append("")
    
    # Summary statistics
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    quality_scores = [r["quality_score"] for r in results]
    similarity_scores = [r["avg_similarity"] for r in results]
    diversity_scores = [r["diversity_score"] for r in results]
    
    report_lines.append(f"Quality Score - Min: {min(quality_scores):.2f}, "
                       f"Max: {max(quality_scores):.2f}, "
                       f"Mean: {np.mean(quality_scores):.2f}, "
                       f"Std: {np.std(quality_scores):.2f}")
    
    report_lines.append(f"Similarity - Min: {min(similarity_scores):.4f}, "
                       f"Max: {max(similarity_scores):.4f}, "
                       f"Mean: {np.mean(similarity_scores):.4f}, "
                       f"Std: {np.std(similarity_scores):.4f}")
    
    report_lines.append(f"Diversity - Min: {min(diversity_scores):.2f}, "
                       f"Max: {max(diversity_scores):.2f}, "
                       f"Mean: {np.mean(diversity_scores):.2f}, "
                       f"Std: {np.std(diversity_scores):.2f}")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if avg_quality >= 80:
        report_lines.append("[OK] Retrieval quality is EXCELLENT!")
        report_lines.append("  The system is retrieving highly relevant documents.")
    elif avg_quality >= 70:
        report_lines.append("[OK] Retrieval quality is GOOD.")
        report_lines.append("  The system performs well but could be optimized.")
        if avg_diversity < 60:
            report_lines.append("  - Consider improving result diversity")
    elif avg_quality >= 60:
        report_lines.append("[WARN] Retrieval quality is FAIR.")
        report_lines.append("  Consider the following improvements:")
        if avg_similarity < 0.5:
            report_lines.append("  - Results similarity is low; check embedding model")
        if avg_diversity < 40:
            report_lines.append("  - Results lack diversity; check hybrid retrieval balance")
    else:
        report_lines.append("[ERROR] Retrieval quality needs significant improvement.")
        report_lines.append("  - Review FAISS index creation and embedding model")
        report_lines.append("  - Ensure sufficient indexed documents")
        report_lines.append("  - Check data preprocessing and chunking strategy")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def save_evaluation_results(results: List[Dict], report: str):
    """Save evaluation results to JSON and text file."""
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = OUTPUTS_DIR / "retrieval_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save text report
    report_path = OUTPUTS_DIR / "retrieval_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"[SAVE] Evaluation results saved to: {json_path}")
    print(f"[SAVE] Evaluation report saved to: {report_path}")
    
    return json_path, report_path


def main():
    """Run retrieval quality evaluation."""
    
    print("=" * 80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)
    print("")
    
    # Load FAISS index and chunks
    print("Loading FAISS index and embeddings...")
    index, embeddings, chunks = load_vector_store()
    
    if index is None or chunks is None:
        print("ERROR: FAISS index not found!")
        print("  Please run: python src/batch_extractor.py")
        return
    
    print(f"[OK] Loaded FAISS index with {len(chunks)} chunks")
    print("")
    
    # Load paper metadata
    metadata = load_paper_metadata()
    print(f"[OK] Loaded metadata for {len(metadata)} papers")
    print("")
    
    # Run evaluation
    print("Running retrieval evaluation...")
    print("-" * 80)
    print("")
    
    results = []
    for idx, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"Evaluating query {idx}/{len(SAMPLE_QUERIES)}: {query}")
        
        result = evaluate_query(query, index, embeddings, chunks, top_k=5)
        results.append(result)
        
        print(f"  Quality Score: {result['quality_score']:.2f}/100")
        print(f"  Retrieved: {result['unique_papers']} unique papers")
        print(f"  Avg Similarity: {result['avg_similarity']:.4f}")
        print("")
    
    # Generate report
    print("-" * 80)
    print("Generating evaluation report...")
    report = generate_evaluation_report(results)
    
    # Save results
    print("")
    save_evaluation_results(results, report)
    
    # Print report
    print("")
    print(report)
    
    # Print report
    print("")
    print(report)
    
    return results, report


if __name__ == "__main__":
    main()
