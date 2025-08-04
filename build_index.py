# build_index.py — FINAL VERSION (OCR-safe, metadata-ready)

import os
import json
import faiss
import numpy as np
from utils.chunker import chunk_all
from utils.embedder import embed_chunks
from utils.indexer import create_faiss_index

# --- Constants ---
DATA_DIR = "data"
INDEX_DIR = "faiss_index"
CHUNK_JSON = "chunks.json"

# --- Step 1: Smart Chunking ---
print("🔍 Chunking documents from:", DATA_DIR)
chunks = chunk_all(DATA_DIR)
print(f"🧩 Total Chunks Created: {len(chunks)}")

# Validate chunk presence
if not chunks:
    raise ValueError("❌ No chunks created. Check your OCR or input files.")

# --- Step 2: Save Chunk File with Metadata ---
print(f"💾 Saving chunks to {CHUNK_JSON}")
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)


# --- Step 3: Embedding ---
print("🔗 Embedding all chunks using SentenceTransformer...")
embeddings = embed_chunks(chunks)
if len(embeddings) != len(chunks):
    raise RuntimeError("❌ Embedding count mismatch with chunks.")

# --- Step 4: Build FAISS Index ---
print("📊 Building FAISS index with dimension:", len(embeddings[0]))
create_faiss_index(np.array(embeddings).astype("float32"), chunks, INDEX_DIR)

print("✅ FAISS Index successfully built and saved at:", INDEX_DIR)
