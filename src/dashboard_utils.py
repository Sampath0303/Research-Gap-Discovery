"""
Utility functions for Research Gap Discovery Engine Dashboard.
Handles data loading, caching, exports, and analytics preparation.
"""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

from src.gap_summarize import parse_gap_response


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PAPERS_DIR = BASE_DIR / "data" / "papers"
RESEARCH_GAPS_PATH = OUTPUTS_DIR / "research_gaps.json"


def _read_json_file(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_gap_record(
    record: Dict[str, Any]
) -> Dict[str, Any]:

    limitations = record.get(
        "limitations",
        []
    )

    if not isinstance(
        limitations,
        list
    ):
        limitations = []

    theme = str(
        record.get(
            "theme",
            ""
        )
    ).strip()

    research_gap = str(
        record.get(
            "research_gap",
            ""
        )
    ).strip()

    potential_idea = str(
        record.get(
            "potential_idea",
            ""
        )
    ).strip()

    if (
        research_gap
        and (
            not theme
            or not potential_idea
        )
    ):
        parsed = parse_gap_response(
            research_gap
        )

        theme = (
            theme
            or parsed["theme"]
        )

        potential_idea = (
            potential_idea
            or parsed[
                "potential_idea"
            ]
        )

        research_gap = (
            parsed[
                "research_gap"
            ]
        )

    return {
        "cluster":
            int(
                record.get(
                    "cluster",
                    0
                )
            ),

        "rank":
            int(
                record.get(
                    "rank",
                    999
                )
            ),

        "theme":
            theme
            or
            "Emerging Research Opportunity",

        "limitations":
            [
                item
                for item
                in limitations
                if isinstance(
                    item,
                    str
                )
                and item.strip()
            ],

        "research_gap":
            research_gap
            or
            "Research gap details are unavailable.",

        "potential_idea":
            potential_idea
            or
            "Potential idea details are unavailable.",

        "supporting_papers":
            record.get(
                "supporting_papers",
                []
            ),

        "cluster_size":
            int(
                record.get(
                    "cluster_size",
                    0
                )
            ),

        "paper_count":
            int(
                record.get(
                    "paper_count",
                    0
                )
            ),

        "score":
            float(
                record.get(
                    "score",
                    0
                )
            ),

        "evidence_score":
            float(
                record.get(
                    "evidence_score",
                    0
                )
            ),
    }

# ============================================================================
# CACHE DECORATORS & DATA LOADING
# ============================================================================


@st.cache_data
@st.cache_data
def load_paper_outputs():

    papers_data = {}

    EXCLUDED_FILES = {
        "research_gaps.json",
        "paper_registry.json",
        "gap_cache.json"
    }

    if not OUTPUTS_DIR.exists():
        return papers_data

    for file_path in sorted(
        OUTPUTS_DIR.glob("*.json")
    ):

        if file_path.name in EXCLUDED_FILES:
            continue

        try:
            data = _read_json_file(
                file_path
            )

        except (
            OSError,
            json.JSONDecodeError
        ):
            continue

        if isinstance(data, dict):

            papers_data[
                file_path.stem
            ] = data

    return papers_data

@st.cache_data
def load_json_outputs() -> Dict[str, Dict[str, Any]]:
    """Backward-compatible alias for paper outputs."""
    return load_paper_outputs()


@st.cache_data
def load_research_gaps() -> List[Dict[str, Any]]:
    """Load structured research gaps from outputs/research_gaps.json."""
    if not RESEARCH_GAPS_PATH.exists():
        return []

    try:
        payload = _read_json_file(RESEARCH_GAPS_PATH)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    return [_normalize_gap_record(record) for record in payload if isinstance(record, dict)]


@st.cache_data
def load_all_limitations() -> List[Dict[str, str]]:
    """Load and aggregate all limitations from paper JSON outputs."""
    all_limitations: List[Dict[str, str]] = []

    for paper_name, data in load_paper_outputs().items():
        limitations = data.get("limitations", [])
        if not isinstance(limitations, list):
            continue

        for limitation in limitations:
            if isinstance(limitation, str) and limitation.strip():
                all_limitations.append(
                    {
                        "paper": paper_name,
                        "limitation": limitation.strip(),
                    }
                )

    return all_limitations


@st.cache_resource
def build_faiss_index() -> Tuple[Any, np.ndarray | None, List | None]:
    """
    Build FAISS index from all PDF chunks.
    Returns: (index, embeddings, chunks)
    """
    try:
        from src.chunker import create_chunks
        from src.pdf_loader import load_all_pdfs
        from src.vector_store import (create_vector_store,load_vector_store)

        index, embeddings, stored_chunks = (
            load_vector_store()
        )

        if index is not None:

            print(
                "Using saved FAISS index"
            )

            return (
                index,
                embeddings,
                stored_chunks
            )

        docs = load_all_pdfs(
            str(PAPERS_DIR)
        )

        chunks = create_chunks(
            docs
        )

        if not chunks:
            return None, None, None

        if index is not None:

            print(
                "Using saved FAISS index"
            )

            return (
                index,
                embeddings,
                stored_chunks
            )

        print(
            "Creating new FAISS index"
        )

        index, embeddings, chunk_objects = (
            create_vector_store(
                chunks
            )
        )

        return (
            index,
            embeddings,
            stored_chunks
        )
    except Exception:
        return None, None, None


def get_cluster_analysis() -> Dict[int, Dict[str, Any]]:
    """Get cluster analysis from structured research gap output."""
    clusters: Dict[int, Dict[str, Any]] = {}
    for gap in load_research_gaps():
        clusters[gap["cluster"]] = {
            "limitations": gap["limitations"],
            "papers": set(),
            "theme": gap["theme"],
            "research_gap": gap["research_gap"],
            "research_idea": gap["potential_idea"],
        }
    return clusters


def generate_cluster_themes(clusters: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Return cluster data in the structure expected by the dashboard."""
    return clusters


# ============================================================================
# STATISTICS
# ============================================================================


def get_statistics() -> Dict[str, int]:
    """Calculate dashboard statistics."""
    paper_outputs = load_paper_outputs()
    all_limitations = load_all_limitations()
    research_gaps = load_research_gaps()

    return {
        "total_papers": len(paper_outputs),
        "total_limitations": len(all_limitations),
        "total_clusters": len(research_gaps),
        "total_research_gaps": len(research_gaps),
    }


# ============================================================================
# SEARCH & RAG
# ============================================================================


def perform_semantic_search(query: str, top_k: int = 10, bm25_weight: float = 0.6, faiss_weight: float = 0.4):
    """
    Perform hybrid semantic search (BM25 + FAISS) and generate RAG answer.
    
    Combines lexical (BM25) and semantic (FAISS) retrieval with optimized weights for research papers.
    BM25-preferred (0.6/0.4) for better technical term matching.
    Uses local fallback if Gemini API is unavailable.

    Args:
        query: Search query string
        top_k: Number of top results to return (default 10 for better context)
        bm25_weight: Weight for BM25 lexical relevance (default 0.6 for technical terms)
        faiss_weight: Weight for FAISS semantic relevance (default 0.4 for concepts)

    Returns:
        retrieved_chunks : List[Dict] with content, paper, page, score
        answer : str - RAG-generated answer or error message
    """
    try:
        from src.rag import generate_answer
        from src.hybrid_search import search_hybrid

        index, embeddings, chunks = build_faiss_index()
        if index is None or chunks is None:
            return [], "FAISS index could not be built. Check whether PDFs exist in data/papers."

        # Perform hybrid search with BM25-preferred weights for technical research papers
        # BM25 (0.6): Captures technical terms, method names, specific architectures
        # FAISS (0.4): Captures semantic meaning and concept relationships
        retrieved_chunks = search_hybrid(
            query,
            index,
            embeddings,
            chunks,
            k=top_k,
            bm25_weight=bm25_weight,
            faiss_weight=faiss_weight
        )
        
        if not retrieved_chunks:
            return [], "No relevant documents found."

        try:
            answer = generate_answer(query, retrieved_chunks)
        except Exception as gemini_error:
            answer = (
                "Gemini API is currently unavailable.\n\n"
                f"Reason: {gemini_error}\n\n"
                f"Hybrid search (BM25 + FAISS) still retrieved {len(retrieved_chunks)} relevant document chunks."
            )

        return retrieved_chunks, answer
    except Exception as error:
        return [], f"Error during hybrid search: {error}"


# ============================================================================
# DATA AGGREGATION HELPERS
# ============================================================================


def get_limitations_by_paper() -> Dict[str, List[str]]:
    """Get limitations grouped by paper name."""
    result: Dict[str, List[str]] = {}

    for paper_name, data in load_paper_outputs().items():
        limitations = data.get("limitations", [])
        result[paper_name] = [
            limitation.strip()
            for limitation in limitations
            if isinstance(limitation, str) and limitation.strip()
        ]

    return result


def get_paper_metadata(paper_name: str) -> Dict[str, Any]:
    """Get complete metadata for a single paper."""
    return load_paper_outputs().get(paper_name, {})


def get_research_gaps_csv() -> str:
    """Convert structured research gaps to CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "cluster",
            "theme",
            "research_gap",
            "potential_idea",
            "limitations_count",
            "limitations",
        ],
    )
    writer.writeheader()

    for gap in load_research_gaps():
        writer.writerow(
            {
                "cluster": gap["cluster"],
                "theme": gap["theme"],
                "research_gap": gap["research_gap"],
                "potential_idea": gap["potential_idea"],
                "limitations_count": len(gap["limitations"]),
                "limitations": " | ".join(gap["limitations"]),
            }
        )

    return output.getvalue()


def get_analytics_data() -> Dict[str, Any]:
    """Prepare data for analytics visualizations."""
    limitations_by_paper = get_limitations_by_paper()
    research_gaps = load_research_gaps()

    lim_per_paper = {
        paper_name: len(limitations)
        for paper_name, limitations in limitations_by_paper.items()
    }

    cluster_sizes = {
        f"Cluster {gap['cluster']}": len(gap["limitations"])
        for gap in research_gaps
    }
    

    theme_labels = {
        gap["theme"]: len(gap["limitations"])
        for gap in research_gaps
    }

    gap_records = [
        {
            "cluster": gap["cluster"],
            "theme": gap["theme"],
            "limitations_count": len(gap["limitations"]),
            "research_gap_length": len(gap["research_gap"].split()),
            "potential_idea_length": len(gap["potential_idea"].split()),
        }
        for gap in research_gaps
    ]
    
    

    return {
        "lim_per_paper": lim_per_paper,
        "cluster_sizes": cluster_sizes,
        "theme_labels": theme_labels,
        "gap_records": gap_records,
        "total_clusters": len(research_gaps),
        "total_research_gaps": len(research_gaps),
        "evidence_scores": {
            gap["theme"]: gap["evidence_score"]
            for gap in research_gaps
        },
    }


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars with ellipsis."""
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def format_limitation(limitation: str, max_lines: int = 3) -> str:
    """Format limitation for display."""
    lines = [segment.strip() for segment in limitation.split(".") if segment.strip()]
    result = ". ".join(lines[:max_lines])
    if len(lines) > max_lines:
        result += "..."
    return result
