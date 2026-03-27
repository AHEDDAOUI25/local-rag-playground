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

Decide which tool should handle the user's request.

Available tools:
- RETRIEVE -> use when the user is asking for a factual answer from the local document knowledge base
- SUMMARIZE -> use when the user is asking for a summary or overview of one or more topics from the local knowledge base
- COMPARE -> use when the user is asking for differences, similarities, or a comparison between concepts from the local knowledge base
- DIRECT -> use when the user is asking for a broad conversational, motivational, or general explanation that does not require the local docs

Rules:
- Reply with only one word: RETRIEVE, SUMMARIZE, COMPARE, or DIRECT
- Use COMPARE when the user says things like compare, difference, similar, versus, vs
- Use SUMMARIZE when the user asks for a summary or overview
- Use RETRIEVE for factual lookups from the local docs
- Use DIRECT for general questions that do not need the docs

Question:
{query}

Decision:
"""

    decision = generate_with_ollama(prompt).strip().upper()

    if "COMPARE" in decision:
        return "COMPARE"
    if "SUMMARIZE" in decision:
        return "SUMMARIZE"
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


def summarize_with_retrieval(query, records):
    results = search(query, records, top_k=4, min_score=0.30)

    if not results:
        return "No strong matching context was found to summarize."

    context = "\n\n".join(
        [f"[Source: {r['source']}, Chunk: {r['chunk_id']}]\n{r['chunk']}" for r in results]
    )

    prompt = f"""
You are an Applied AI knowledge assistant.

Your task is to summarize the relevant information from the provided context.

Instructions:
- Write a concise, clear summary.
- Use only the provided context.
- Do not invent information.
- End with a short Sources section listing the source file and chunk IDs used.

User request:
{query}

Context:
{context}

Summary:
"""

    return generate_with_ollama(prompt)


def compare_with_retrieval(query, records):
    results = search(query, records, top_k=5, min_score=0.25)

    if not results:
        return "No strong matching context was found to compare those topics."

    context = "\n\n".join(
        [f"[Source: {r['source']}, Chunk: {r['chunk_id']}]\n{r['chunk']}" for r in results]
    )

    prompt = f"""
You are an Applied AI knowledge assistant.

Your task is to compare the concepts requested by the user using only the provided context.

Instructions:
- Clearly explain the main differences and/or similarities.
- Keep the answer structured and concise.
- Use only the provided context.
- Do not invent information.
- End with a short Sources section listing the source file and chunk IDs used.

User request:
{query}

Context:
{context}

Comparison:
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
        elif decision == "SUMMARIZE":
            print("-> Using summarization tool\n")
            answer = summarize_with_retrieval(query, all_records)
        elif decision == "COMPARE":
            print("-> Using compare tool\n")
            answer = compare_with_retrieval(query, all_records)
        else:
            print("-> Answering directly\n")
            answer = answer_directly(query)
        print("=" * 60)
        print(answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()