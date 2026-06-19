from sklearn.cluster import KMeans
from src.limitation_extractor import all_limitations
from src.models import embedding_model


texts = [
    item["limitation"]
    for item in all_limitations
]

embeddings = embedding_model.encode(
    texts
)

n_clusters = 5

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42
)

labels = kmeans.fit_predict(
    embeddings
)

for i in range(n_clusters):

    print("\n" + "=" * 50)
    print(f"CLUSTER {i}")
    print("=" * 50)

    for idx, label in enumerate(labels):

        if label == i:

            print(
                "-",
                texts[idx]
            )
