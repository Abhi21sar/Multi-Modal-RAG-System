# app/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the text embedding model
text_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast

def get_text_embedding(text):
    return text_model.encode(text, convert_to_numpy=True)

def embed_documents(docs):
    """
    docs: list of dicts from ingestion.py, each having 'content', 'filename', 'filepath'
    Returns a list of tuples: (embedding_vector, metadata_dict)
    """
    embedded_docs = []
    for doc in docs:
        embedding = get_text_embedding(doc['content'])
        metadata = {
            "filename": doc['filename'],
            "filepath": doc['filepath'],
            "content": doc['content']
        }
        embedded_docs.append((embedding, metadata))
    return embedded_docs