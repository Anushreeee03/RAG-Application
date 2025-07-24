# build_index.py (final version with OCR + JSON fix)

import os
import faiss
import numpy as np
import json
from utils.chunker import chunk_all
from utils.embedder import embed_chunks
from utils.indexer import create_faiss_index

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

print("🔍 Chunking documents...")
chunks = chunk_all(DATA_DIR)
print(f"🧩 Total Chunks: {len(chunks)}")

# Save chunks to JSON (optional inspection)
with open("chunks.json") as f:
    chunks = json.load(f)


# Prepare for embedding
texts = [c["text"] for c in chunks]
embeddings = embed_chunks(chunks)  # pass full chunk list for traceability

print("📊 Building FAISS index...")
create_faiss_index(np.array(embeddings).astype("float32"), chunks, INDEX_DIR)
print("✅ Index built successfully!")
