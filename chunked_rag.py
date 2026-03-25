from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

document = """
Microsoft Fabric is a unified analytics platform that brings together data engineering,
data integration, data warehousing, data science, real-time analytics, and business intelligence.

OneLake is the unified data lake for Microsoft Fabric. It allows teams to store data once
and use it across multiple analytics experiences.

RAG stands for Retrieval-Augmented Generation. It improves question answering by retrieving
relevant external information before generating a response.

Vector databases store embeddings, which are numerical representations of meaning.
They help systems search by semantic similarity instead of exact keyword matching.

Azure Machine Learning helps teams train, deploy, and manage machine learning models.
It supports experiment tracking, model management, and endpoint deployment.

Semantic Kernel is a Microsoft framework for building AI agents. It helps connect LLMs
with tools, plugins, planners, and memory.

LangChain is another framework often used to build RAG applications, chains, and agents.
"""

def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_by_sentences(sentences, chunk_size=2, overlap=1):
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
    return chunks

# Step 1: split and chunk
sentences = split_into_sentences(document)
chunks = chunk_by_sentences(sentences, chunk_size=2, overlap=1)

print("\nSentence-Based Chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}\n")

# Step 2: embed chunks
chunk_embeddings = model.encode(chunks)

# Step 3: ask a question
query = "What tools helps manage machine learning models?"
query_embedding = model.encode([query])

# Step 4: compare query to chunks
scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

# Step 5: retrieve best chunks
top_k = 2
top_indices = np.argsort(scores)[::-1][:top_k]

print("\nTop matching chunks:\n")
retrieved_chunks = []
for idx in top_indices:
    print(f"Score: {scores[idx]:.4f}")
    print(chunks[idx])
    print()
    retrieved_chunks.append(chunks[idx])

# Step 6: build retrieved context
context = "\n".join(retrieved_chunks)

print("=" * 60)
print("QUESTION:")
print(query)
print("\nRETRIEVED CONTEXT:")
print(context)
print("=" * 60)