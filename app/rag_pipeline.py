import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import ollama
from chat_history import add_document_record
import logging
import time
import os

os.makedirs("../processed/logs", exist_ok=True)

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

if not logger.handlers:  # avoid duplicate handlers when uvicorn reloads
    file_handler = logging.FileHandler("../processed/logs/app.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())  # also show in the terminal

# Load everything once, when this file is imported 

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Connecting to Chroma...")
client = chromadb.PersistentClient(path="../processed/chroma_db")
collection = client.get_or_create_collection(name="chunks_1000")

print("Loading chunks...")
with open("../processed/chunks_1000.json") as f:
    chunks = json.load(f)

print("Building BM25 index...")
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_chunks = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

print("Loading page lookup for citations...")
with open("../processed/extracted_docs.json") as f:
    extracted_docs = json.load(f)
page_lookup = {doc["filename"]: doc["pages"] for doc in extracted_docs}

print("RAG pipeline ready.")

# Core functions 

def rank_normalize(scores):
    scores = np.array(scores)
    n = len(scores)
    ranks = np.argsort(np.argsort(-scores))
    return 1 - (ranks / (n - 1))

def hybrid_search(query, top_k=5, alpha=0.5):
    query_embedding = model.encode([query])
    all_results = collection.query(query_embeddings=query_embedding.tolist(), n_results=len(chunks))

    semantic_distances = {cid: dist for cid, dist in zip(all_results["ids"][0], all_results["distances"][0])}
    semantic_scores_ordered = [1 - semantic_distances.get(c["chunk_id"], 1) for c in chunks]

    tokenized_query = tokenize(query)
    bm25_scores_ordered = bm25.get_scores(tokenized_query)

    sem_norm = rank_normalize(semantic_scores_ordered)
    bm25_norm = rank_normalize(bm25_scores_ordered)
    combined = alpha * sem_norm + (1 - alpha) * bm25_norm

    top_indices = np.argsort(combined)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def rerank(query, candidates, top_k=3):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = score
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

def find_page(filename, chunk_text, sample_len=80):
    sample = chunk_text[:sample_len].strip()
    pages = page_lookup.get(filename, [])
    for page in pages:
        if sample[:40] in page["text"]:
            return page["page_num"]
    return None

def format_citation(chunk):
    page = find_page(chunk["filename"], chunk["text"])
    snippet = chunk["text"][:150].strip() + "..."
    page_str = f"page {page}" if page else "page unknown"
    return f"{chunk['filename']} ({page_str}): \"{snippet}\""

def get_context(query, top_k=3):
    candidates = hybrid_search(query, top_k=15, alpha=0.5)
    best_chunks = rerank(query, candidates, top_k=top_k)
    context_text = ""
    for i, c in enumerate(best_chunks):
        context_text += f"[Source {i+1}: {c['filename']}]\n{c['text']}\n\n"
    return context_text, best_chunks

def rag_answer(query, top_k=3, history=None):
    start = time.time()
    try:
        context_text, sources = get_context(query, top_k=top_k)
        retrieval_time = time.time() - start

        history_text = ""
        if history:
            for h in history[-3:]:
                history_text += f"Previous question: {h['question']}\nPrevious answer: {h['answer']}\n\n"

        prompt = f"""Answer the question using ONLY the context below. Give a complete, detailed answer in 2-4 sentences, explaining the concept clearly, not just naming it.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not use any outside knowledge.

{f"Conversation so far:{chr(10)}{history_text}" if history_text else ""}
Context:
{context_text}

Question: {query}

Answer:"""

        llm_start = time.time()
        response = ollama.chat(model="llama3.2:3b", messages=[
            {"role": "user", "content": prompt}
        ])
        llm_time = time.time() - llm_start
        answer = response["message"]["content"]

        citations = [format_citation(s) for s in sources]

        logger.info(
            f"question={query!r} | retrieval={retrieval_time:.2f}s | llm={llm_time:.2f}s "
            f"| total={time.time() - start:.2f}s | sources={len(sources)} "
            f"| idk={'I don' in answer and 'know' in answer}"
        )
        return {"answer": answer, "sources": citations}

    except Exception as e:
        logger.error(f"question={query!r} | error={e}")
        raise

       
import fitz  # pymupdf
import os

UPLOAD_DIR = "../PDFs"

def process_new_pdf(file_path, filename, uploaded_by="unknown"):    
    """Extract, chunk, embed, and index a single newly uploaded PDF."""

    # 1. Extract text
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        pages.append({"page_num": page_num + 1, "text": page.get_text()})
    doc.close()
    full_text = "\n".join(p["text"] for p in pages)

    # 2. Clean
    cleaned = re.sub(r"-\n", "", full_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned).strip()

    # 3. Chunk
    new_chunks = []
    start = 0
    chunk_size, overlap = 1000, 200
    i = 0
    while start < len(cleaned):
        piece = cleaned[start:start + chunk_size]
        new_chunks.append({
            "chunk_id": f"{filename}_{i}",
            "filename": filename,
            "chunk_index": i,
            "text": piece,
        })
        start += chunk_size - overlap
        i += 1

    # 4. Embed
    texts = [c["text"] for c in new_chunks]
    new_embeddings = model.encode(texts)

    # 5. Add to Chroma
    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=new_embeddings.tolist(),
        documents=texts,
        metadatas=[{"filename": c["filename"], "chunk_index": c["chunk_index"]} for c in new_chunks],
    )

    # 6. Add to in-memory chunks list + rebuild BM25 (BM25 has no "add", must rebuild)
    global chunks, bm25, tokenized_chunks, page_lookup
    chunks.extend(new_chunks)
    tokenized_chunks.extend([tokenize(c["text"]) for c in new_chunks])
    bm25 = BM25Okapi(tokenized_chunks)

    # 7. Add to page lookup, for citations
    page_lookup[filename] = pages

    add_document_record(filename, len(new_chunks), uploaded_by)

    logger.info(f"upload | file={filename} | chunks={len(new_chunks)} | by={uploaded_by}")

    return len(new_chunks)