from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Hollister sells casual clothing and fashion apparel.",
    "Azure ML helps deploy and manage machine learning models.",
    "Vector databases store embeddings for semantic retrieval.",
    "Microsoft Fabric supports lakehouse-style data engineering.",
    "Dogs are loyal and friendly animals."
]

query = "tools for storing embeddings and searching by meaning"

# Embed documents and query
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Compute similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Rank results
ranked_indices = np.argsort(scores)[::-1]

print(f"\nQuery: {query}\n")
print("Top matches:\n")

for i in ranked_indices:
    print(f"Score: {scores[i]:.4f} | Document: {documents[i]}")