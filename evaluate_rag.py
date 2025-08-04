import json
import numpy as np
import time
import evaluate
from sentence_transformers import SentenceTransformer
from utils.indexer import load_faiss_index
from utils.groq_client import query_groq
from rag_utils import normalize_answer, fuzzy_match

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def evaluate_rag(eval_file, index_dir="faiss_index"):
    # Load FAISS + Chunks
    index, chunks = load_faiss_index(index_dir)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load questions
    with open(eval_file, "r") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} evaluation records.")

    predictions = []
    ground_truths = []
    latencies = []
    results = []

    for item in records:
        question = item["question"]
        true_answer = item["answer"]

        # Embedding & FAISS search
        q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_vec, k=5)
        matches = [chunks[i] for i in indices[0] if i != -1 and scores[0][list(indices[0]).index(i)] > 0.2]

        context = "\n\n".join([f"{i+1}. {m['text']}" for i, m in enumerate(matches)])

        prompt = f"""
Answer the question ONLY using the context.

Context:
{context}

Question: {question}
Answer:"""

        start = time.time()
        pred = query_groq(prompt, "")
        latency = round(time.time() - start, 2)

        predictions.append(pred.strip())
        ground_truths.append(true_answer.strip())
        latencies.append(latency)

        results.append({
            "question": question,
            "prediction": pred.strip(),
            "ground_truth": true_answer.strip(),
            "latency_sec": latency,
            "sources": list({m["source"] for m in matches}),
            "fuzzy_score": fuzzy_match(pred, true_answer)
        })

    # Evaluation Metrics
    exact_match = np.mean([
        normalize_answer(p) == normalize_answer(t) for p, t in zip(predictions, ground_truths)
    ])
    fuzzy_avg = np.mean([
        fuzzy_match(p, t) for p, t in zip(predictions, ground_truths)
    ])
    rouge_score = rouge.compute(predictions=predictions, references=ground_truths)
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in ground_truths])

    metrics = {
        "exact_match": round(exact_match, 3),
        "fuzzy_avg": round(fuzzy_avg, 3),
        "rougeL": round(rouge_score["rougeL"], 3),
        "bleu": round(bleu_score["bleu"], 3),
        "avg_latency_sec": round(np.mean(latencies), 2)
    }

    with open("evaluation_report.json", "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics
