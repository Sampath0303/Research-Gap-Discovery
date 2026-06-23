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

    for rank, idx in enumerate(indices[0]):

        if idx < 0:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "content":
                    chunk.page_content,

                "paper":
                    chunk.metadata.get(
                        "source_file",
                        "Unknown"
                    ),

                "page":
                    chunk.metadata.get(
                        "page",
                        "?"
                    ),

                "score":
                    float(
                        distances[0][rank]
                    )
            }
        )

    return results