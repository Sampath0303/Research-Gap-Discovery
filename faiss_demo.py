from src.models import embedding_model
import faiss
import numpy as np

# Load transformer model
model = embedding_model

# Our mini knowledge base
documents = [
    "Transformers require large datasets",
    "CNNs are effective for image classification",
    "Reinforcement learning learns by rewards",
    "Large training datasets improve transformer performance",
    "LSTMs are useful for sequence modeling"
]

# Convert documents into embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

print(f"Number of documents stored: {index.ntotal}")

# User query
query = "Need more data for transformers"

query_embedding = model.encode([query])

# Find top 2 similar documents
k = 2

distances, indices = index.search(
    np.array(query_embedding).astype("float32"),
    k
)

print("\nQuery:")
print(query)

print("\nTop Matches:")

for i in indices[0]:
    print("-", documents[i])