import faiss
import numpy as np

from src.models import embedding_model


def create_vector_store(chunks):
    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings, chunks
