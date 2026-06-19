from src.models import embedding_model
from sklearn.metrics.pairwise import cosine_similarity

model = embedding_model

sentences = [
    "Transformers require large datasets",
    "Large amounts of data are needed by transformers",
    "I love playing cricket"
]

embeddings = model.encode(sentences)

sim_1_2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

sim_1_3 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]

print(f"Similarity(1,2): {sim_1_2:.4f}")
print(f"Similarity(1,3): {sim_1_3:.4f}")