import _bootstrap
from src.models import embedding_model

print("Loading model...")

model = embedding_model

sentences = [
    "Transformers require large datasets",
    "Large amounts of data are needed by transformers",
    "I love playing cricket"
]

embeddings = model.encode(sentences)

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nFirst 10 values of first embedding:")
print(embeddings[0][:10])
