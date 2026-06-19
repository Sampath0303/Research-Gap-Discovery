import numpy as np

from src.models import embedding_model


def search_documents(
    query,
    index,
    chunks,
    k=5
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:
        results.append(
            chunks[idx]
        )

    return results
