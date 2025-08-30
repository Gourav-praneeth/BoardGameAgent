import os
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "../data"
INDEX_DIR = "../embeddings"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index(game_file):
    game_name = os.path.splitext(os.path.basename(game_file))[0]
    index_path = os.path.join(INDEX_DIR, f"{game_name}_index.pkl")

    if game_file.endswith(".pdf"):
        text = extract_text_from_pdf(game_file)
    else:
        with open(game_file, "r", encoding="utf-8") as f:
            text = f.read()

    chunks = chunk_text(text)
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    with open(index_path, "wb") as f:
        pickle.dump((index, chunks), f)

    print(f"✅ Built index for {game_name} -> {index_path}")

if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf") or file.endswith(".txt"):
            build_index(os.path.join(DATA_DIR, file))