from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import requests

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(sentences, chunk_size=2, overlap=1):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break

    return chunks


def build_index(text, chunk_size=2, overlap=1):
    sentences = split_into_sentences(text)
    chunks = chunk_by_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
    embeddings = model.encode(chunks)
    return chunks, embeddings


def search(query, chunks, embeddings, top_k=2):
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


def build_grounded_answer(query, results):
    return generate_with_ollama(query, results)


def generate_with_ollama(query, results, model="llama3.2:3b"):
    context = "\n".join([r["chunk"] for r in results])

    prompt = f"""
You are answering only from the provided context.

Question:
{query}

Context:
{context}

Instructions:
- Answer clearly and directly.
- Use only the context.
- If the context is insufficient, say so.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    return data["response"]








def main():
    filepath = "knowledge_base.txt"
    document = load_text_file(filepath)

    chunks, embeddings = build_index(document)

    print("\nRAG Retrieval App (type 'exit' to quit)\n")
    print(f"Loaded file: {filepath}")
    print(f"Number of chunks: {len(chunks)}\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.\n")
            continue

        results = search(query, chunks, embeddings, top_k=2)

        print("\nTop matches:\n")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(r["chunk"])
            print()

        print("=" * 60)
        print(build_grounded_answer(query, results))
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()