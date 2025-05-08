"""
retriever.py

A document retriever class using SentenceTransformers for semantic embeddings
and FAISS for vector similarity search. Supports adding, querying, saving, and loading.
"""

import os
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=500):
        """
        Initializes the retriever with a sentence transformer and FAISS index.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_size = chunk_size
        self.doc_chunks = []

    def _chunk_text(self, text):
        """
        Splits text into overlapping chunks of approximately chunk_size words.
        """
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - 50)  # Ensure positive step size
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        # Ensure at least one chunk for very short texts
        if not chunks and len(words) > 0:
            chunks = [" ".join(words)]

        return chunks

    def add_documents(self, docs):
        """
        Takes a list of documents, chunks and encodes them, and builds a FAISS index.
        """
        print(f"Adding {len(docs)} documents to the retriever. {self.chunk_size}")
        all_chunks = []
        for doc in docs:
            chunks = self._chunk_text(doc)
            if not chunks:
                continue  # skip empty documents
            self.doc_chunks.extend(chunks)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No valid chunks to encode. Check document input or chunking logic.")

        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def query(self, query, top_k=3):
        """
        Returns top_k chunks relevant to the query.
        """
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), top_k)
        return [(self.doc_chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

    def save(self, path='retriever_store'):
        """
        Saves the FAISS index and document chunks.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.doc_chunks, f)

    def load(self, path='retriever_store'):
        """
        Loads the FAISS index and document chunks.
        """
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'chunks.pkl'), 'rb') as f:
            self.doc_chunks = pickle.load(f)

if __name__ == "__main__":
    retriever = Retriever(chunk_size=100)
    sample_doc = "Salahuddin Ayubi was a Muslim military leader who recaptured Jerusalem from the Crusaders."
    retriever.add_documents([sample_doc])

    query = "Who took back Jerusalem?"
    results = retriever.query(query)
    for text, score in results:
        print(f"Score: {score:.4f} | Result: {text}")