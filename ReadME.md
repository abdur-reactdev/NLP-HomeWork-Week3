# 🧠 NLP Homework Week 3 – FAISS Document Retriever

This project is a lightweight semantic search engine using [FAISS](https://github.com/facebookresearch/faiss) and [SentenceTransformers](https://www.sbert.net/) for natural language document retrieval.

The `Retriever` class supports document chunking, indexing, semantic querying, and persistence through saving/loading.

---

## 🔧 Features

- Semantic embedding using SentenceTransformers
- Fast nearest-neighbor search with FAISS
- Built-in document chunking
- Easy interface for adding, querying, saving, and loading
- Supports `.txt`, `.md`, and `.pdf` files

---

## 📂 Project Structure

```bash
NLP-HomeWork-Week3/
│
├── retriever.py # Core Retriever class (chunking, indexing, query, save/load)
├── usage_example.ipynb # Example Colab/Notebook for testing retriever
├── test_retriever.py # Automated unit test (retrieves known answer)
├── documents/ # Input docs (e.g., PDF, TXT)
├── README.md # Project description and usage guide
└── retriever_store/ # Saved FAISS index and chunks (created at runtime)
```
