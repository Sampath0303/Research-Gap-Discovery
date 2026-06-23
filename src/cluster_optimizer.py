from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(
    embeddings,
    min_k: int = 2,
    max_k: int = 6
) -> int:
    """
    Find the optimal number of clusters using
    silhouette score.
    """

    n_samples = len(embeddings)

    if n_samples < 3:
        return 1

    max_k = min(
        max_k,
        n_samples - 1
    )

    best_k = min_k
    best_score = -1

    print("\nFinding optimal cluster count...")

    for k in range(
        min_k,
        max_k + 1
    ):

        try:

            print(f"Testing k={k}")

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(
                embeddings
            )

            score = silhouette_score(
                embeddings,
                labels
            )

            print(
                f"k={k}, score={score:.4f}"
            )

            if score > best_score:

                best_score = score
                best_k = k

        except Exception as e:

            print(
                f"k={k} failed: {e}"
            )

            continue

    print(
        f"\nBest K selected: {best_k}"
    )

    return best_k