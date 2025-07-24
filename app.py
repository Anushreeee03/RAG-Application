# app.py (final interactive Gemini-style chatbot with source-aware answers)

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.indexer import load_faiss_index
from utils.groq_client import query_groq
import faiss

st.set_page_config(page_title="📊 Quarterly Financial Report", layout="centered", initial_sidebar_state="collapsed")

# === Load FAISS index and Sentence Transformer ===
@st.cache_resource
def load():
    index, chunks = load_faiss_index("faiss_index")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder

index, chunks, embedder = load()

# === UI ===
st.title("📊 Quarterly Financial Report Bot")
st.markdown("Ask questions from your uploaded financial reports.\n\nAccurate answers from real data. No hallucinations.")

query = st.text_area("🔍 Enter your question:", height=10)

if st.button("🚀 Answer") and query.strip():
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=5)

    matches = [chunks[i] for i in indices[0] if i != -1 and scores[0][list(indices[0]).index(i)] > 0.2]

    if not matches:
        st.warning("⚠️ I couldn't find enough relevant data to answer this question.")
    else:
        context = "\n\n".join([f"{i+1}. {m['text']}" for i, m in enumerate(matches)])

        fewshot = f"""
Example 1:
Context:
1. Net Income: ₹38,354 Cr
2. Revenue: ₹195,772 Cr

Question: What was the net profit in Q4 FY22?
Answer: The net profit was ₹38,354 Cr.

Example 2:
Context:
1. Net Income: 241.8
2. EPS: 1.22

Question: What was the EPS in Q1 FY2025?
Answer: The EPS was 1.22 USD.

Now answer based only on this context:

Context:
{context}

Question: {query}
Answer:"""

        answer = query_groq(context=fewshot, question="")

        st.success(answer)

        st.markdown("---")
        st.markdown("### 📚 Sources Used:")
        for m in matches:
            st.markdown(f"**📄 {m['source']}**\n> {m['text'][:300]}{'...' if len(m['text']) > 300 else ''}")

        st.markdown("\n---\n*Powered by FAISS + MiniLM + GROQ API* 🌐")
