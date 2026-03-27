from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import requests
import os


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    

def load_all_text_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append({
                    "source": filename,
                    "text": f.read()
                })
    return documents


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


def build_index(text, source_name, chunk_size=2, overlap=1):
    sentences = split_into_sentences(text)
    chunks = chunk_by_sentences(sentences, chunk_size=chunk_size, overlap=overlap)
    embeddings = model.encode(chunks)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": i,
            "source": source_name,
            "chunk": chunk,
            "embedding": embeddings[i]
        })
    return records





#old search 
""""
def search(query, records, top_k=2):
    query_embedding = model.encode([query])

    chunk_embeddings = np.array([r["embedding"] for r in records])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk_id": records[idx]["chunk_id"],
            "source": records[idx]["source"],
            "chunk": records[idx]["chunk"]
        })

    return results
""" 
#new search that filters and sorts with a higher threshold to meet query relevance
def search(query, records, top_k=3, min_score=0.35):
    query_embedding = model.encode([query])

    chunk_embeddings = np.array([r["embedding"] for r in records])
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

    ranked_indices = np.argsort(scores)[::-1]

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])

        if score < min_score:
            continue

        results.append({
            "score": score,
            "chunk_id": records[idx]["chunk_id"],
            "source": records[idx]["source"],
            "chunk": records[idx]["chunk"]
        })

        if len(results) == top_k:
            break

    return results



def build_grounded_answer(query, results):
    return generate_with_ollama(query, results)

def generate_with_ollama(query, results, model="llama3.2:1b"):
    if not results:
        return "No strong matching context was found for that question."

    context = "\n\n".join(
        [f"[Source: {r['source']}, Chunk: {r['chunk_id']}]\n{r['chunk']}" for r in results]
    )

    prompt = f"""
Answer the user's question using only the context below.

If the answer is present in the context, answer directly and confidently.
Do not say the information is insufficient unless the context truly does not contain the answer.
After the answer, include a short Sources section listing the source and chunk IDs you relied on.
Keep the answer concise and clear.

Question:
{query}

Context:
{context}

Answer:
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


#first main that runs one txt file for the RAG system
"""""

def main():
    filepath = "knowledge_base.txt"
    document = load_text_file(filepath)

    records = build_index(document, source_name=filepath)
    print("\nRAG Retrieval App (type 'exit' to quit)\n")
    print(f"Loaded file: {filepath}")
    print(f"Number of chunks: {len(records)}\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.\n")
            continue

        results = search(query, records, top_k=2)

        print("\nTop matches:\n")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(f"Source: {r['source']} | Chunk ID: {r['chunk_id']}")
            print(r["chunk"])
            print()

        print("=" * 60)
        print(build_grounded_answer(query, results))
        print("=" * 60)
        print()
"""""
def main():
    folder_path = "docs"
    documents = load_all_text_files(folder_path)

    all_records = []

    for doc in documents:
        records = build_index(doc["text"], source_name=doc["source"])
        all_records.extend(records)

    print("\nLocal RAG App (type 'exit' to quit)\n")
    print(f"Loaded folder: {folder_path}")
    print(f"Files loaded: {len(documents)}")
    print(f"Total chunks: {len(all_records)}\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.\n")
            continue

        results = search(query, all_records, top_k=3)
        if not results:
            print("\nNo strong matches found.\n")
            print("=" * 60)
            print("No strong matching context was found for that question.")
            print("=" * 60)
            print()
            continue

        print("\nTop matches:\n")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(f"Source: {r['source']} | Chunk ID: {r['chunk_id']}")
            print(r["chunk"])
            print()

        print("=" * 60)
        print(generate_with_ollama(query, results))
        print("=" * 60)
        print()



if __name__ == "__main__":
    main()