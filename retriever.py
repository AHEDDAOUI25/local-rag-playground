from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(sentences, chunk_size=2, overlap=1):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break

    return chunks


def build_chunk_index(text, chunk_size=2, overlap=1):
    sentences = split_into_sentences(text)
    chunks = chunk_by_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
    embeddings = model.encode(chunks)
    return chunks, embeddings


def search_chunks(query, chunks, embeddings, top_k=2):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk": chunks[idx]
        })
    return results


if __name__ == "__main__":
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

    chunks, embeddings = build_chunk_index(document, chunk_size=2, overlap=1)

    print("\nChunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"{i}: {chunk}\n")

    query = "What does Semantic Kernel do?"
    results = search_chunks(query, chunks, embeddings, top_k=2)

    print(f"\nQuery: {query}\n")
    print("Top results:\n")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(r["chunk"])
        print()