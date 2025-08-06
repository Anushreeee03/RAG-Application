# evaluate.py

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.groq_client import query_groq
from rag_utils import normalize_answer, fuzzy_match

def evaluate_rag(eval_file="evaluation_dataset.jsonl", chunks_path="chunks.json"):
    print("✅ Starting Evaluation...")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks.")

    with open(eval_file, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"✅ Loaded {len(records)} evaluation questions.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    predictions, ground_truths, fuzzy_scores = [], [], []
    total_latency = 0

    for item in records:
        question = item.get("question", "").strip()
        true_answer = item.get("answer", "").strip()

        if not question or not true_answer:
            continue

        q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
        chunk_vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True).astype("float32")
        scores = np.dot(chunk_vecs, q_vec.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:5]
        top_chunks = [chunks[idx] for idx in top_k_indices if scores[idx] > 0.2]

        if not top_chunks:
            prediction = "I don’t know based on the provided data."
            latency = 0
        else:
            context = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(top_chunks)])
            prompt = f"""
Answer ONLY using a short numeric or currency value from this context.

Context:
{context}

Question: {question}
Answer:"""

            start = time.time()
            prediction = query_groq(prompt, "").strip()
            latency = time.time() - start
            total_latency += latency

        fuzzy_score = fuzzy_match(prediction, true_answer)
        predictions.append(prediction)
        ground_truths.append(true_answer)
        fuzzy_scores.append(fuzzy_score)

    y_true = [1] * len(predictions)
    y_pred = [1 if score >= 0.85 else 0 for score in fuzzy_scores]

    avg_latency = round(total_latency / len(predictions), 3) if predictions else 0.0

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "avg_latency_sec": avg_latency
    }

    results = [
        {
            "question": q,
            "prediction": p,
            "ground_truth": t,
            "fuzzy_score": s
        }
        for q, p, t, s in zip([r["question"] for r in records], predictions, ground_truths, fuzzy_scores)
    ]

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    print("\n✅ Evaluation Complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate_rag()
