# app.py (Final version with sidebar, smart prompts, no repeat, insight mode)

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.indexer import load_faiss_index
from utils.groq_client import query_groq
import faiss

# === Page Config ===
st.set_page_config(
    page_title="📈 FinBot: Quarterly Report Buddy",
    page_icon="🤖",
    layout="wide"
)

# === Load FAISS index and embedder ===
@st.cache_resource
def load():
    index, chunks = load_faiss_index("faiss_index")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, embedder

index, chunks, embedder = load()

# === Custom CSS ===
st.markdown("""
    <style>
    html { color-scheme: light dark; }
    .big-title { font-size: 2.4em !important; font-weight: bold; color: var(--title-color); }
    .chat-bubble { border-radius: 10px; padding: 15px; margin: 10px 0; color: var(--text-color); }
    .bot-msg { background-color: var(--bot-bg); }
    .source-box { background-color: var(--source-bg); border-left: 4px solid #1a73e8; padding: 10px; margin: 8px 0; border-radius: 6px; color: var(--text-color); }
    :root {
        --title-color: #1a73e8;
        --text-color: #000000;
        --bot-bg: #fff8e1;
        --source-bg: #f1f1f1;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --title-color: #66b2ff;
            --text-color: #ffffff;
            --bot-bg: #333300;
            --source-bg: #1e1e1e;
        }
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=80)  # Optional logo/icon
    st.title("🤖 FinBot")
    st.markdown("**Your Quarterly Report Assistant**")
    st.markdown("**Suggested Questions:**")
    example_questions = [
        "What was TCS’s net profit in Q4 FY22?",
        "What was Infosys’s revenue in Q1 FY2025",
        "What was the Net Income in Q1 FY2025 in Data Company LTD?",
        "What was the average diluted shares outstanding in Q1 2025"
    ]
    for q in example_questions:
        if st.button(q):
            st.session_state['query'] = q

    insight_mode = st.checkbox("💡 Enable Insight Mode", value=True)
    st.markdown("---")
    st.caption("Version 1.0 | Made with ❤️ using FAISS + GROQ")

# === Main Area ===
st.markdown('<div class="big-title">📈 FinBot</div>', unsafe_allow_html=True)
st.markdown("**Ask any question about your quarterly reports.**")

# === Input ===
query = st.text_area("🔍 Your Question:", value=st.session_state.get('query', ''), height=80)

# === Answer ===
if st.button("🚀 Get Answer") and query.strip():
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=5)

    matches = [chunks[i] for i in indices[0] if i != -1 and scores[0][list(indices[0]).index(i)] > 0.2]

    if not matches:
        st.warning("⚠️ Couldn't find enough relevant data. Try rephrasing!")
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

Now answer ONLY from this context:

Context:
{context}

Question: {query}
Answer:"""

        answer = query_groq(context=fewshot, question="")

        st.markdown(f"""
            <div class="chat-bubble bot-msg">🤖FinBot:<br>{answer}</div>
        """, unsafe_allow_html=True)

        # === Insight Mode ===
        if insight_mode:
            insight_prompt = f"""
Here is the answer: {answer}

1. Provide a **one-line summary**.
2. Suggest **one smart follow-up question** to explore further.
"""
            insight = query_groq(context=insight_prompt, question="")
            st.info(f"💡 **Insight Mode:**\n\n{insight}")

        st.markdown("### 📚 **Sources Used:**")
        for m in matches:
            st.markdown(f"""
                <div class="source-box">
                <strong>📄 {m['source']}</strong><br>
                {m['text'][:300]}{'...' if len(m['text']) > 300 else ''}
                </div>
            """, unsafe_allow_html=True)

        st.session_state['query'] = ''  # clear for next