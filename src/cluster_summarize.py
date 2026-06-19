import json
from pathlib import Path

from sklearn.cluster import KMeans
from src.limitation_extractor import all_limitations
from src.models import embedding_model
from src.rag import client

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "research_gaps.json"

texts = [
    item["limitation"]
    for item in all_limitations
]

embeddings = embedding_model.encode(texts)

n_clusters = 3

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

labels = kmeans.fit_predict(
    embeddings
)

cluster_map = {}

for idx, label in enumerate(labels):

    if label not in cluster_map:
        cluster_map[label] = []

    cluster_map[label].append(
        texts[idx]
    )

results = []

for cluster_id, limitations in cluster_map.items():

    joined_text = "\n".join(
        limitations
    )

    prompt = f"""
You are an AI research analyst.

Analyze these limitations from research papers.

Limitations:
{joined_text}

Return:

1. Theme
2. Research Gap
3. Novel Research Project Idea

Be concise.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result = {
        "cluster": cluster_id,
        "limitations": limitations,
        "analysis": response.text
    }

    results.append(result)

    print("\n")
    print("=" * 60)
    print(f"CLUSTER {cluster_id}")
    print("=" * 60)
    print(response.text)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
)
