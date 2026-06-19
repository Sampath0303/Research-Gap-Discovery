import json
from pathlib import Path
from typing import Dict, List

from sklearn.cluster import KMeans

from src.gap_summarize import generate_gap
from src.limitation_extractor import load_all_limitations
from src.models import embedding_model


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "research_gaps.json"


def build_fallback_gap(limitations: List[str]) -> Dict[str, str]:
    return {
        "theme": "Limited Evidence Cluster",
        "research_gap": "Too few related limitations were found in this cluster to justify a strong cross-paper research gap.",
        "potential_idea": "Expand the paper set for this topic and validate whether the limitation pattern persists across additional studies.",
    }


def generate_research_gaps() -> List[Dict]:
    all_limitations = load_all_limitations()
    texts = [item["limitation"] for item in all_limitations]

    if not texts:
        return []

    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    n_clusters = min(5, len(texts))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(embeddings)

    cluster_map: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(texts[idx])

    research_gaps = []
    for cluster_id, limitations in sorted(cluster_map.items()):
        gap_payload = (
            generate_gap(limitations)
            if len(limitations) >= 3
            else build_fallback_gap(limitations)
        )

        research_gaps.append(
            {
                "cluster": cluster_id,
                "theme": gap_payload["theme"],
                "limitations": limitations,
                "research_gap": gap_payload["research_gap"],
                "potential_idea": gap_payload["potential_idea"],
            }
        )

    return research_gaps


def save_research_gaps(research_gaps: List[Dict]) -> Path:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(research_gaps, file, indent=4, ensure_ascii=False)
    return OUTPUT_FILE


def main():
    research_gaps = generate_research_gaps()
    save_research_gaps(research_gaps)


if __name__ == "__main__":
    main()
