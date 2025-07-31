
# 📊 FinBot: A Retrieval-Augmented Generation (RAG) System for Financial Reports

FinBot is a GenAI-powered assistant built as part of a NexTurn internship project. It answers natural language queries about quarterly financial reports from real-world companies such as Infosys, TCS, Reliance, and Netflix.


## 🚀 Objective

To design and deploy a Retrieval-Augmented Generation (RAG) pipeline that:

- Ingests multi-format financial documents (PDF, TXT, CSV, JSON)
- Extracts and embeds text data with metadata
- Answers questions using vector similarity and LLM (LLaMA3 via GROQ)
- Returns accurate, short financial responses with source references
- Includes an evaluation mode for accuracy tracking

---

## 🧠 Architecture Overview

```text
 User Query
     ↓
Query Encoder (MiniLM)
     ↓
FAISS (IVF Index)
     ↓
Top-k Chunks
     ↓
LLM (GROQ LLaMA3-8B)
     ↓
 Answer + Insight
````

---

## 📁 Folder Structure

```
├── app.py                  # Streamlit UI
├── build_index.py          # Document processing and index creation
├── evaluate_rag.py         # Offline evaluation script
├── evaluation_dataset.jsonl# Evaluation Q&A pairs
├── chunks.json             # All parsed document chunks + metadata
├── faiss_index/            # FAISS index and chunk.pkl
├── data/                   # Folder with 10 source documents
│   ├── *.pdf / *.txt / *.json / *.csv
├── utils/
│   ├── chunker.py          # Chunking logic for all formats
│   ├── embedder.py         # MiniLM embedding logic
│   ├── indexer.py          # FAISS indexing
│   ├── groq_client.py      # GROQ LLM call with retry
├── rag_utils.py            # Evaluation utilities (fuzzy match, normalization)
```


## 📄 Data Sources

10 financial documents across these companies:

* **TCS** – Q4 FY22 (PDF)
* **Infosys** – Q1 FY25 (TXT)
* **Reliance** – Annual summary (TXT)
* **Netflix** – Quarterly earnings (CSV)
* **DataCompany Ltd** – Structured financials (JSON)
* **Quarterly Report Definitions & Concepts** – TXT


## 🔨 Key Technologies

| Component       | Tool / Library                   |
| --------------- | -------------------------------- |
| Chunking        | PyMuPDF, Pandas, OCR (Tesseract) |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB       | FAISS (`IndexIVFFlat`)           |
| LLM Inference   | GROQ API (LLaMA3-8B)             |
| UI              | Streamlit                        |
| Evaluation      | `sklearn` + fuzzy string match   |

---

## 💡 Prompt Engineering

### Answer Prompt

```
ONLY return short numeric/currency answer.

Context:
{top_k_chunks}

Q: {user_question}
A:
```

### Insight Mode Prompt

```
Give a 1-line summary and follow-up question for: {answer}
```


## 🧪 Evaluation

Evaluation is run using:

```bash
python evaluate_rag.py
```

### Metrics Used:

* Accuracy
* Precision
* Recall
* F1-Score
* Avg Latency
* Fuzzy Match (>= 0.85 = correct)

### Example Result

```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "avg_latency_sec": 0.66
}
```


## 🎯 Features

* Parses PDFs, CSVs, TXTs, and JSONs into paragraph chunks
* Performs semantic search using FAISS
* Uses LLaMA3 (via GROQ) for short, numeric financial responses
* Shows relevant source documents for transparency
* Includes “Insight Mode” for contextual suggestions


##  How to Run

1. Place documents in the `data/` folder
2. Run `build_index.py` to chunk, embed, and index documents
3. Start the app:

```bash
streamlit run app.py
```


## 📈 Sample Questions

* What was Infosys’s net income in Q1?
* What was TCS’s revenue in Q4 FY22?
* What was Reliance’s total revenue?
* What was the EPS for Data Company Ltd?
* What was the headcount of TCS?

## 📌 Future Enhancements

* Cross-encoder re-ranking
* Real-time document upload
* Better company/quarter metadata extraction
* Hybrid retrieval (BM25 + FAISS)
* Feedback-based improvement

## 🙌 Acknowledgments

* Project completed as part of **NexTurn Internship – GenAI Engineering Track**
* Thanks to mentors and reviewers for guidance

## 👤 Author

**Anushree Sathyan**
📧 [anushree.sathyan@nexturn.com](mailto:anushree.sathyan@nexturn.com)
🗂️ Topic: Quarterly Financial Reports using RAG

