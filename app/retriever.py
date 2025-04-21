# app/retriever.py

import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim, index_path="vector_db/index.faiss", metadata_path="vector_db/metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add_embeddings(self, embedded_docs):
        vectors = [vec for vec, _ in embedded_docs]
        metadatas = [meta for _, meta in embedded_docs]

        vectors_np = np.vstack(vectors).astype("float32")
        self.index.add(vectors_np)
        self.metadata.extend(metadatas)

    def search(self, query_vector, top_k=5):
        D, I = self.index.search(np.array([query_vector]).astype("float32"), top_k)
        results = [self.metadata[i] for i in I[0]]
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)