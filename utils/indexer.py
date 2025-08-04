import os
import faiss
import pickle
import numpy as np

def create_faiss_index(embeddings, chunks, index_dir):
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 100)
    index.train(embeddings)
    index.add(embeddings)


    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(index_dir):
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
