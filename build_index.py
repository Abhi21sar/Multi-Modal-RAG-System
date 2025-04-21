# build_index.py

from app.ingestion import extract_from_folder
from app.embedder import embed_documents
from app.retriever import VectorStore
import os

# Make sure vector_db folder exists
os.makedirs("vector_db", exist_ok=True)

# Step 1: Extract content from files
docs = extract_from_folder("data/")

# Step 2: Generate embeddings
embedded_docs = embed_documents(docs)

# Step 3: Create FAISS index and save
dim = embedded_docs[0][0].shape[0]
store = VectorStore(dim=dim)
store.add_embeddings(embedded_docs)
store.save()

print("✅ FAISS index and metadata saved to vector_db/")