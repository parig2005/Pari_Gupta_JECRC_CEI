# 📄 Chat with PDF

An AI-powered chatbot that lets you upload a PDF and ask questions about it using Retrieval-Augmented Generation (RAG).

## 🛠️ Built With
- **Streamlit** — web interface
- **LangChain** — RAG pipeline
- **OpenAI** — language model
- **ChromaDB** — vector store

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run chat_with_pdf.py
```

## 🔑 Setup
Add your OpenAI API key in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-key-here"
```

## 💡 How It Works
1. PDF is loaded and split into chunks
2. Chunks are embedded and stored in ChromaDB
3. Your question is matched to relevant chunks
4. OpenAI generates an answer based on those chunks
