import streamlit as st
import numpy as np
import json
import time
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.indexer import load_faiss_index
from utils.groq_client import query_groq
from rag_utils import fuzzy_match, get_eval_metrics

st.set_page_config(page_title="📈 FinBot", page_icon="🤖", layout="wide")

@st.cache_resource
def load():
    index, chunks = load_faiss_index("faiss_index")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder

index, chunks, embedder = load()

# === Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)
    st.title("🤖 FinBot")
    st.markdown("**Suggested Questions:**")
    example_questions = [
        "What was Infosys’s net income in Q1?",
        "What was TCS’s revenue?",
        "What was Reliance’s total revenue?",
    ]
    for i, q in enumerate(example_questions):
        if st.button(q, key=f"example_{i}"):
            st.session_state["query"] = q
    insight_mode = st.checkbox("💡 Enable Insight Mode", value=True)
    show_eval = st.checkbox("📊 Show Evaluation Dashboard")
    st.caption("FinBot v1.0 | FAISS + GROQ")

# === Query Input
st.markdown('<h1 style="font-size: 2.4em;">📈 FinBot</h1>', unsafe_allow_html=True)
query = st.text_area("🔍 Your Question:", value=st.session_state.get("query", ""), height=80)

# === Answering
if st.button("🚀 Get Answer") and query.strip():
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    chunk_vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True).astype("float32")
    idx = faiss.IndexFlatIP(chunk_vecs.shape[1])
    idx.add(chunk_vecs)
    scores, indices = idx.search(q_vec, k=5)

    matches = [chunks[i] for i in indices[0] if i != -1 and scores[0][list(indices[0]).index(i)] > 0.3]

    if not matches:
        st.warning("⚠️ No relevant context found.")
    else:
        context = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(matches)])
        prompt = f"""
ONLY return short numeric/currency answer.

Context:
{context}

Q: {query}
A:"""

        start_time = time.time()
        answer = query_groq(prompt).strip()
        latency = round(time.time() - start_time, 2)

        st.markdown(f"""
        <div style='
            padding: 16px;
            border-radius: 8px;
            background-color: rgba(240,240,240,0.1);
            border: 1px solid rgba(200,200,200,0.3);
            color: inherit;
        '>
        <strong>🤖 FinBot:</strong><br>{answer}
        </div>
        """, unsafe_allow_html=True)

        # 🔍 Insight Mode
        if insight_mode and "i don't know" not in answer.lower():
            insight_prompt = f"""
Given the financial answer '{answer}' from the context below, summarize its significance in one line, and suggest a meaningful follow-up question.

Context:
{context}
"""
            insight = query_groq(insight_prompt)
            st.info(f"💡 {insight}")

        # ✅ Inline Evaluation
        eval_path = Path("evaluation_dataset.jsonl")
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = [json.loads(line) for line in f if line.strip()]
                eval_lookup = {r["question"].strip().lower(): r["answer"] for r in eval_data}

            if query.lower().strip() in eval_lookup:
                true_ans = eval_lookup[query.lower().strip()]
                metrics = get_eval_metrics(answer, true_ans)

                st.subheader("📏 Evaluation Metrics (per query)")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Fuzzy", metrics["fuzzy"])
                c2.metric("Accuracy", metrics["accuracy"])
                c3.metric("Precision", metrics["precision"])
                c4.metric("Recall", metrics["recall"])
                c5.metric("F1 Score", metrics["f1"])
                st.success(f"✅ Ground Truth: `{true_ans}`")

        st.markdown("### 📚 Sources Used:")
        for m in matches:
            st.markdown(f"""
            <div style='
                padding: 10px;
                margin: 8px 0;
                border-left: 4px solid #1a73e8;
                border-radius: 5px;
                background-color: rgba(255,255,255,0.05);
                color: inherit;
            '>
            <strong>📄 {m['source']}</strong><br>{m['text'][:300]}{'...' if len(m['text']) > 300 else ''}
            </div>
            """, unsafe_allow_html=True)

# === Evaluation Dashboard
if show_eval:
    st.markdown("## 📊 Evaluation Dashboard")
    if st.button("🔁 Run Evaluation Now"):
        eval_path = Path("evaluation_dataset.jsonl")
        if not eval_path.exists():
            st.warning("⚠️ evaluation_dataset.jsonl not found.")
        else:
            with st.spinner("Running evaluation..."):
                import subprocess
                subprocess.run(["python", "evaluate_rag.py"])

            report_path = Path("evaluation_report.json")
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                    metrics = report["metrics"]

                st.subheader("📈 Evaluation Metrics")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", metrics["accuracy"])
                c2.metric("Precision", metrics["precision"])
                c3.metric("Recall", metrics["recall"])
                c4.metric("F1 Score", metrics["f1"])
                c5.metric("Avg Latency (s)", metrics["avg_latency_sec"])

                st.success("✅ Evaluation Complete")
                st.code(json.dumps(metrics, indent=2))
            else:
                st.error("❌ evaluation_report.json not found.")
