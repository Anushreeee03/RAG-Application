import os
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import json

def chunk_txt(file_path):
    with open(file_path, 'r') as f:
        paragraphs = f.read().split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]

def chunk_csv(file_path):
    df = pd.read_csv(file_path)
    return [row.to_json() for _, row in df.iterrows()]

def chunk_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [json.dumps(obj) for obj in data]
    else:
        return [json.dumps(data)]

def chunk_pdf_images(file_path):
    images = convert_from_path(file_path)
    chunks = []
    for img in images:
        text = pytesseract.image_to_string(img)
        paragraphs = text.split('\n\n')
        chunks.extend([p.strip() for p in paragraphs if p.strip()])
    return chunks

def chunk_all(data_folder):
    all_chunks = []
    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)
        if filename.endswith('.txt'):
            all_chunks.extend(chunk_txt(path))
        elif filename.endswith('.csv'):
            all_chunks.extend(chunk_csv(path))
        elif filename.endswith('.json'):
            all_chunks.extend(chunk_json(path))
        elif filename.endswith('.pdf'):
            all_chunks.extend(chunk_pdf_images(path))
    return all_chunks
