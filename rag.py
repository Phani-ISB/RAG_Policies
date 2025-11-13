# -*- coding: utf-8 -*-
"""RAG with policies — Streamlit App"""

import os
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import tempfile

# --- Prevent watchdog errors on Streamlit Cloud ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

# --- Streamlit Page Config ---
st.set_page_config(page_title="📚 RAG Chatbot with LlamaIndex + Google API", layout="wide")
st.title("💬 RAG Chatbot (LlamaIndex + Google Generative AI)")
st.caption("Upload up to 2 PDFs. Uses SentenceTransformers + Google Generative AI for RAG.")

# --- Ensure API key exists ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize LLM once
llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=os.environ["GOOGLE_API_KEY"])

# --- Directory to persist index ---
INDEX_DIR = Path("./storage")
INDEX_DIR.mkdir(exist_ok=True)

# --- Helper: Load PDF and create Documents ---
def load_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        docs.append(Document(text=text, metadata={"source": file.name}))
    return docs

# --- Build or Load Index ---
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_or_create_index(docs):
    persist_dir = "./storage"
    if os.path.exists(persist_dir) and any(Path(persist_dir).iterdir()):
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        # Create new index
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        index.storage_context.persist(persist_dir=persist_dir)
    return index

# --- Streamlit App Interface ---
uploaded_files = st.file_uploader("Upload up to 2 PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner("📄 Processing and indexing documents..."):
        docs = load_pdfs(uploaded_files)
        index = get_or_create_index(docs)
        st.success("✅ Index built successfully!")

    # --- Create query engine using Google LLM ---
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

    # --- Chat Interface ---
    st.subheader("💭 Ask a question about your PDFs")
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("🤔 Generating answer..."):
            response = query_engine.query(user_query)
            st.markdown("### 🧠 Answer:")
            st.write(response.response)
else:
    st.info("👆 Please upload 1 or 2 PDFs to start.")
