import json
from pathlib import Path
from typing import Dict, List

from sklearn.cluster import KMeans

from src.gap_summarize import generate_gap
from src.limitation_extractor import load_all_limitations
from src.models import embedding_model
from src.cluster_optimizer import find_optimal_k
from src.metadata_index import MetadataIndex


BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "research_gaps.json"
)


def build_fallback_gap(
    limitations: List[str]
) -> Dict[str, str]:

    return {
        "theme": "Limited Evidence Cluster",

        "research_gap": (
            "Too few related limitations "
            "were found in this cluster "
            "to justify a strong "
            "cross-paper research gap."
        ),

        "potential_idea": (
            "Expand the paper set for "
            "this topic and validate "
            "whether the limitation "
            "pattern persists across "
            "additional studies."
        ),
    }


def calculate_advanced_scores(
    cluster_size: int,
    paper_count: int,
    theme_frequency: int,
    max_cluster_size: int,
    max_paper_count: int,
    max_theme_frequency: int,
    existing_evidence_score: float = 0.0
) -> Dict[str, float]:
    """
    Calculate advanced research gap scores using multi-factor weighting.
    
    Scores:
    - frequency_score (0.4): Number of limitations (cluster_size)
    - diversity_score (0.3): Number of supporting papers
    - novelty_score (0.2): Inverse theme frequency (uniqueness)
    - evidence_score (0.1): Confidence metric
    
    All scores normalized to 0-1 range.
    
    Args:
        cluster_size: Number of limitations in cluster
        paper_count: Number of supporting papers
        theme_frequency: How many gaps have this theme
        max_cluster_size: Maximum cluster size in dataset
        max_paper_count: Maximum paper count in dataset
        max_theme_frequency: Maximum theme frequency in dataset
        existing_evidence_score: Pre-calculated confidence metric (0-1)
        
    Returns:
        Dict with frequency_score, diversity_score, novelty_score, evidence_score, final_score
    """
    
    # Normalize frequency_score: proportion of max cluster size
    # Higher cluster size = more instances of limitation = higher frequency
    frequency_score = cluster_size / max_cluster_size if max_cluster_size > 0 else 0
    
    # Normalize diversity_score: proportion of max paper count
    # More papers = more diversity in evidence = higher diversity
    diversity_score = paper_count / max_paper_count if max_paper_count > 0 else 0
    
    # Calculate novelty_score: inverse theme frequency (uniqueness)
    # Inverse: 1 / (1 + theme_frequency)
    # Lower frequency = more novel = higher score
    novelty_score = 1.0 / (1.0 + theme_frequency) if max_theme_frequency > 0 else 0.5
    novelty_score = novelty_score / (1.0 / 1.0) if max_theme_frequency > 0 else 0.5  # Normalize to 0-1
    
    # Evidence score: provided metric or derive from normalized cluster/paper ratio
    evidence_score = existing_evidence_score
    if evidence_score == 0.0:
        # Derive from cluster quality
        evidence_score = (0.6 * frequency_score + 0.4 * diversity_score)
    
    # Calculate final score with weights
    # 0.4 * frequency + 0.3 * diversity + 0.2 * novelty + 0.1 * evidence
    final_score = (
        0.4 * frequency_score +
        0.3 * diversity_score +
        0.2 * novelty_score +
        0.1 * evidence_score
    )
    
    return {
        'frequency_score': round(frequency_score, 3),
        'diversity_score': round(diversity_score, 3),
        'novelty_score': round(novelty_score, 3),
        'evidence_score': round(evidence_score, 3),
        'final_score': round(final_score, 3),
    }


def calculate_score(
    cluster_size: int,
    paper_count: int
) -> float:
    """Legacy function for backward compatibility."""
    diversity_score = paper_count
    score = (
        0.5 * cluster_size
        + 0.3 * paper_count
        + 0.2 * diversity_score
    )
    return round(score, 2)


def generate_research_gaps() -> List[Dict]:

    all_limitations = (
        load_all_limitations()
    )

    texts = [
        item["limitation"]
        for item in all_limitations
    ]

    if not texts:
        return []

    embeddings = (
        embedding_model.encode(
            texts,
            show_progress_bar=False
        )
    )

    n_clusters = find_optimal_k(
        embeddings,
        min_k=2,
        max_k=6
    )

    print(
        f"Optimal clusters determined: "
        f"{n_clusters}"
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = (
        kmeans.fit_predict(
            embeddings
        )
    )

    cluster_map: Dict[
        int,
        List[Dict]
    ] = {}

    for idx, label in enumerate(labels):

        cluster_map.setdefault(
            int(label),
            []
        ).append(
            all_limitations[idx]
        )

    research_gaps = []
    print(
        "\nClustering complete."
    )
    print(
        f"Total clusters: {len(cluster_map)}"
    )

    for (
        cluster_id,
        cluster_items
    ) in sorted(
        cluster_map.items()
    ):

        limitation_texts = [
            item["limitation"]
            for item in cluster_items
        ]

        supporting_papers = sorted(
            list(
                {
                    item["paper"]
                    for item in cluster_items
                }
            )
        )

        cluster_size = len(
            limitation_texts
        )

        paper_count = len(
            supporting_papers
        )

        if len(limitation_texts) >= 3:
            print(
                f"\nCluster {cluster_id}: ")

            gap_payload = (
                generate_gap(
                    limitation_texts
                )
            )

        else:

            gap_payload = (
                build_fallback_gap(
                    limitation_texts
                )
            )

        score = calculate_score(
            cluster_size,
            paper_count
        )

        research_gaps.append(
            {
                "cluster": cluster_id,

                "theme":
                    gap_payload["theme"],

                "limitations":
                    limitation_texts,

                "supporting_papers":
                    supporting_papers,

                "cluster_size":
                    cluster_size,

                "paper_count":
                    paper_count,

                "score":
                    score,

                "research_gap":
                    gap_payload[
                        "research_gap"
                    ],

                "potential_idea":
                    gap_payload[
                        "potential_idea"
                    ],
            }
        )

    # Calculate theme frequencies for novelty scoring
    theme_frequency_map = {}
    for gap in research_gaps:
        theme = gap["theme"]
        theme_frequency_map[theme] = (
            theme_frequency_map.get(theme, 0) + 1
        )
    
    max_theme_frequency = (
        max(theme_frequency_map.values())
        if theme_frequency_map else 1
    )

    # Find normalization factors
    max_cluster = max(
        gap["cluster_size"]
        for gap in research_gaps
    ) if research_gaps else 1

    max_papers = max(
        gap["paper_count"]
        for gap in research_gaps
    ) if research_gaps else 1

    # Calculate advanced scores for each gap
    for gap in research_gaps:
        theme_freq = theme_frequency_map.get(
            gap["theme"],
            1
        )
        
        scores = calculate_advanced_scores(
            cluster_size=gap["cluster_size"],
            paper_count=gap["paper_count"],
            theme_frequency=theme_freq,
            max_cluster_size=max_cluster,
            max_paper_count=max_papers,
            max_theme_frequency=max_theme_frequency,
            existing_evidence_score=0.0
        )
        
        # Add score components to gap
        gap["frequency_score"] = scores["frequency_score"]
        gap["diversity_score"] = scores["diversity_score"]
        gap["novelty_score"] = scores["novelty_score"]
        gap["evidence_score"] = scores["evidence_score"]
        gap["final_score"] = scores["final_score"]

    # Sort by final_score (descending)
    research_gaps.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Add rank based on final_score
    for rank, gap in enumerate(
        research_gaps,
        start=1
    ):
        gap["rank"] = rank

    return research_gaps


def save_research_gaps(
    research_gaps: List[Dict]
) -> Path:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            research_gaps,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Add gaps to metadata database
    metadata_index = MetadataIndex()
    metadata_index.initialize_database()

    with metadata_index:
        for gap in research_gaps:
            metadata_index.add_gap(
                theme=gap["theme"],
                cluster_size=gap.get("cluster_size", 0),
                paper_count=gap.get("paper_count", 0),
                score=gap.get("score", 0.0)
            )

    return OUTPUT_FILE


def main():

    research_gaps = (
        generate_research_gaps()
    )

    save_research_gaps(
        research_gaps
    )


if __name__ == "__main__":
    main()