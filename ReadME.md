
```bash
pip install PyPDF2
```

--- 

## 📄 Usage

1. Place your `.txt`, `.md`, or `.pdf` documents in the `documents` folder.
2. Run the script to build the FAISS index:
    ```bash
    python build_index.py
    ```
3. Query the retriever:
    ```bash
    python query_retriever.py "Your search query here"
    ```

--- 

## 📂 File Structure

```
.
├── documents/
├── build_index.py
├── query_retriever.py
└── ReadME.md
```

--- 

## 🧪 Example

```bash
python query_retriever.py "What is FAISS?"
```

--- 

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements.

--- 

## 📜 License

This project is licensed under the MIT License.