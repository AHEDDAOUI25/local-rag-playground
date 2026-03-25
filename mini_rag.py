from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample enterprise-style documents
documents = [
    "Vector databases store embeddings for semantic retrieval.",
    "RAG combines retrieval with text generation to answer questions using external knowledge.",
    "Microsoft Fabric supports lakehouse-style data engineering and analytics.",
    "Azure ML helps deploy and manage machine learning models.",
    "Semantic Kernel is a Microsoft framework for building AI agents and tool-using workflows.",
    "LangChain helps build chains, retrieval pipelines, and agentic applications."
]

# User question
query = "How does RAG help answer questions using company data?"

# Step 1: Create embeddings
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Step 2: Compute similarity
scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Step 3: Rank top matches
top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]

retrieved_docs = []
print(f"\nQuery: {query}\n")
print("Top retrieved documents:\n")

for idx in top_indices:
    print(f"Score: {scores[idx]:.4f} | {documents[idx]}")
    retrieved_docs.append(documents[idx])

# Step 4: Build context block
context = "\n".join(retrieved_docs)

# Step 5: Simple answer generation scaffold
answer = f"""
Question:
{query}

Retrieved Context:
{context}

Draft Answer:
RAG helps answer questions using company data by first retrieving the most relevant information from stored documents,
then using that retrieved context to support a grounded answer. This reduces guessing and makes responses more accurate.
"""

print("\n" + "=" * 60)
print(answer)
print("=" * 60)