import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "../embeddings"
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index(game_name):
    path = os.path.join(INDEX_DIR, f"{game_name}_index.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No index found for {game_name}. Run preprocess.py first.")
    with open(path, "rb") as f:
        index, chunks = pickle.load(f)
    return index, chunks

def search_rules(game_name, query, k=3):
    index, chunks = load_index(game_name)
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = [{"content": chunks[i], "score": float(dist)} for i, dist in zip(indices[0], distances[0])]
    return results