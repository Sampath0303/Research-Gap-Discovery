import faiss
import pickle
import numpy as np
from pathlib import Path

from src.models import embedding_model


BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_FILE = (
    BASE_DIR
    / "outputs"
    / "faiss.index"
)

EMBEDDINGS_FILE = (
    BASE_DIR
    / "outputs"
    / "embeddings.npy"
)

CHUNKS_FILE = (
    BASE_DIR
    / "outputs"
    / "chunks.pkl"
)


def create_vector_store(chunks):

    if not chunks:
        raise ValueError(
            "No chunks available."
        )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.encode(
            texts,
            show_progress_bar=True
        )
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    save_vector_store(
        index,
        embeddings,
        chunks
    )

    return (
        index,
        embeddings,
        chunks
    )


def save_vector_store(
    index,
    embeddings,
    chunks
):

    FAISS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_FILE)
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    with open(
        CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print(
        f"FAISS saved -> {FAISS_FILE}"
    )


def load_vector_store():

    if (
        not FAISS_FILE.exists()
        or
        not EMBEDDINGS_FILE.exists()
        or
        not CHUNKS_FILE.exists()
    ):
        return None, None, None

    index = faiss.read_index(
        str(FAISS_FILE)
    )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    with open(
        CHUNKS_FILE,
        "rb"
    ) as file:

        chunks = pickle.load(
            file
        )

    print(
        f"FAISS loaded -> {FAISS_FILE}"
    )

    return (
        index,
        embeddings,
        chunks
    )


def append_embeddings(new_embeddings, existing_embeddings):
    """
    Append new embeddings to existing embeddings array.
    
    Args:
        new_embeddings: np.ndarray of new embeddings
        existing_embeddings: np.ndarray of existing embeddings
        
    Returns:
        np.ndarray: Combined embeddings
    """
    if existing_embeddings is None or existing_embeddings.shape[0] == 0:
        return new_embeddings.astype("float32")
    
    new_embeddings = new_embeddings.astype("float32")
    combined = np.vstack([existing_embeddings, new_embeddings])
    
    return combined.astype("float32")


def save_updated_index(index, embeddings, chunks):
    """
    Save updated FAISS index, embeddings, and chunks without full rebuild.
    
    Args:
        index: FAISS index object
        embeddings: np.ndarray of embeddings
        chunks: List of chunks
    """
    if index is None or embeddings is None or chunks is None:
        raise ValueError("Invalid data: index, embeddings, or chunks is None")
    
    FAISS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(FAISS_FILE))
    np.save(EMBEDDINGS_FILE, embeddings)
    
    with open(CHUNKS_FILE, "wb") as file:
        pickle.dump(chunks, file)
    
    print(f"FAISS updated -> {FAISS_FILE}")
    print(f"Total vectors: {embeddings.shape[0]}")
    print(f"Total chunks: {len(chunks)}")


def get_index_stats():
    """
    Get statistics about the current FAISS index.
    
    Returns:
        dict: Statistics including ntotal, dimension, chunk_count
    """
    index, embeddings, chunks = load_vector_store()
    
    if index is None:
        return {"ntotal": 0, "dimension": 0, "chunk_count": 0}
    
    return {
        "ntotal": index.ntotal,
        "dimension": embeddings.shape[1],
        "chunk_count": len(chunks) if chunks else 0,
    }