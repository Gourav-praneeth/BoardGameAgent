import os
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# --- Config ---
DATA_DIR = "../data"
INDEX_DIR = "../embeddings"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper functions ---
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
    """Build FAISS index for a new game (PDF or TXT)."""
    game_name = os.path.splitext(os.path.basename(game_file))[0]
    index_path = os.path.join(INDEX_DIR, f"{game_name}_index.pkl")

    if game_file.endswith(".pdf"):
        text = extract_text_from_pdf(game_file)
    else:
        with open(game_file, "r", encoding="utf-8") as f:
            text = f.read()

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    with open(index_path, "wb") as f:
        pickle.dump((index, chunks), f)

    return game_name

def search_rules(query, game, k=3):
    """Search for relevant rules inside selected game."""
    index_path = os.path.join(INDEX_DIR, f"{game}_index.pkl")
    if not os.path.exists(index_path):
        return []

    with open(index_path, "rb") as f:
        index, chunks = pickle.load(f)

    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = [{"content": chunks[i], "score": float(dist)} for i, dist in zip(indices[0], distances[0])]
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="Board Game Rules Agent", page_icon="🎲")

st.title("🎲 Board Game Rules Agent")

# Upload new rulebook
uploaded_file = st.file_uploader("📂 Upload a new game rulebook (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    # Default game name = filename without extension
    default_name = os.path.splitext(uploaded_file.name)[0]

    # Let user edit or confirm the name
    game_name = st.text_input("✏️ Enter game name:", value=default_name)

    if st.button("📥 Save Rulebook"):
        # Save the file with the cleaned game name
        ext = os.path.splitext(uploaded_file.name)[1]
        file_path = os.path.join(DATA_DIR, f"{game_name}{ext}")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Build FAISS index
        build_index(file_path)

        st.success(f"✅ Added {game_name} to the library!")

# List available games
games = [f.replace("_index.pkl", "") for f in os.listdir(INDEX_DIR) if f.endswith(".pkl")]
if not games:
    st.warning("⚠️ No games available yet. Please upload a PDF/TXT rulebook.")
    st.stop()

selected_game = st.selectbox("🎮 Choose a game:", games)

query = st.text_input("❓ Ask a question about the rules:")

if query:
    st.info(f"Searching rules for **{selected_game}**...")

    relevant = search_rules(query, selected_game, k=3)
    if relevant:
        context = "\n\n".join([c["content"] for c in relevant])

        prompt = f"""
Game: {selected_game}
Relevant rules:
{context}

Question: {query}

Answer concisely, with only the game rules (setup, turn structure, win condition, special rules).
Use bullet points if needed. No extra commentary.
"""

        response = model_gemini.generate_content(prompt)
        answer = response.text

        st.markdown("### 📘 Answer:")
        st.write(answer)

        with st.expander("📖 Retrieved rulebook context"):
            st.write(context)

    else:
        st.warning("No relevant rules found. Try rephrasing your question.")