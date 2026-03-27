from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import re
import os

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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
    embeddings = embedding_model.encode(chunks)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": i,
            "source": source_name,
            "chunk": chunk,
            "embedding": embeddings[i]
        })

    return records


def search(query, records, top_k=3, min_score=0.35):
    query_embedding = embedding_model.encode([query])

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


def generate_with_ollama(prompt, model="llama3.2:1b"):
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

def decide_tool(query):
    prompt = f"""
You are a routing controller for an Applied AI assistant.

Your job is to decide whether the user question should use:
- RETRIEVE -> if the question likely needs information from the local document knowledge base
- DIRECT -> if the question can be answered directly without retrieval

Rules:
- Reply with only one word: RETRIEVE or DIRECT
- Use RETRIEVE for questions about topics likely covered in the local docs, such as:
  Microsoft Fabric, OneLake, RAG, Azure ML, Semantic Kernel, LangChain, Power BI, AI agents
- Use DIRECT for broad conversational or motivational questions that do not require the docs

Question:
{query}

Decision:
"""

    decision = generate_with_ollama(prompt).strip().upper()

    if "RETRIEVE" in decision:
        return "RETRIEVE"
    return "DIRECT"


def answer_with_retrieval(query, records):
    results = search(query, records, top_k=3, min_score=0.35)

    if not results:
        return "No strong matching context was found for that question."

    context = "\n\n".join(
        [f"[Source: {r['source']}, Chunk: {r['chunk_id']}]\n{r['chunk']}" for r in results]
    )

    prompt = f"""
You are an Applied AI knowledge assistant.

Answer the user's question using only the provided context.
If the answer is present in the context, answer directly and clearly.
Do not invent information.
After the answer, include a short Sources section listing the source file and chunk IDs used.

Question:
{query}

Context:
{context}

Answer:
"""

    return generate_with_ollama(prompt)


def answer_directly(query):
    prompt = f"""
You are an Applied AI learning assistant.

Answer the following question clearly and concisely.
If the question is broad, give a practical explanation.

Question:
{query}

Answer:
"""
    return generate_with_ollama(prompt)


def main():
    folder_path = "docs"
    documents = load_all_text_files(folder_path)

    all_records = []
    for doc in documents:
        records = build_index(doc["text"], source_name=doc["source"])
        all_records.extend(records)

    print("\nApplied AI Agent (type 'exit' to quit)\n")
    print(f"Loaded folder: {folder_path}")
    print(f"Files loaded: {len(documents)}")
    print(f"Total chunks: {len(all_records)}\n")

    while True:
        query = input("Ask the agent: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.\n")
            continue

        decision = decide_tool(query)

        print("\nAgent decision:")
        if decision == "RETRIEVE":
            print("-> Using retrieval tool\n")
            answer = answer_with_retrieval(query, all_records)
        else:
            print("-> Answering directly\n")
            answer = answer_directly(query)

        print("=" * 60)
        print(answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()