# app/ui.py

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.retriever import VectorStore
from app.generator import generate_answer

# Load vector store
store = VectorStore(dim=384)
store.load()

st.set_page_config(page_title="Multi-Modal RAG Assistant", layout="wide")
st.title("🧠 Multi-Modal RAG AI Assistant")

query = st.text_input("Ask a question related to your documents:", "")

if query:
    with st.spinner("Thinking..."):
        answer, sources = generate_answer(query, store)

    st.markdown("### ✅ AI Answer")
    st.markdown(answer)

    st.markdown("---")
    st.markdown("### 📄 Sources Used")
    for doc in sources:
        with st.expander(doc["filename"]):
            st.markdown(doc["content"][:1500] + "...")