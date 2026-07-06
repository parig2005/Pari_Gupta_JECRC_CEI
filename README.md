# Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) pipeline that answers questions using your own custom documents — PDFs, text files, or a Hugging Face dataset — instead of relying only on a language model's internal, possibly outdated or hallucinated, knowledge.

## Overview

Instead of asking a language model to answer purely from memory, this system:
1. Retrieves the most relevant chunks of text from your documents
2. Adds that text as context to the model's prompt (augmentation)
3. Generates a final answer grounded in the retrieved content

This improves factual accuracy and enables question answering over private or domain-specific data that the model was never trained on.

## Architecture

```
Documents (PDF/TXT/HF dataset)
        │
        ▼
  Document Ingestion  ──►  Text Chunking  ──►  Embedding Creation
                                                       │
                                                       ▼
                                              Vector Database (Chroma)
                                                       │
User Question ──► Query Embedding ──► Context Retrieval ◄┘
                                                       │
                                                       ▼
                                          Answer Generation (LLM)
```

## Tech Stack

| Component        | Tool                                              |
|-------------------|---------------------------------------------------|
| Chunking          | LangChain `RecursiveCharacterTextSplitter`         |
| Embedding model   | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector database   | Chroma (local, file-based, no server required)     |
| LLM (generation)  | Local model via [Ollama](https://ollama.com)       |
| Orchestration     | LangChain                                          |

## Project Structure

```
RAG_Document_Question_Answering/
├── data/
│   ├── raw/              # Put your PDFs / text files here
│   └── sample/           # Example documents for a quick demo
├── notebook/
│   └── RAG_Pipeline.ipynb   # Step-by-step interactive walkthrough
├── src/
│   ├── data_loader.py    # Load PDFs, text files, or HF datasets
│   ├── chunker.py        # Split text into overlapping chunks
│   ├── vector_store.py   # Embeddings + Chroma vector database
│   └── rag.py            # Retrieval + prompt building + generation
├── main.py               # CLI entry point (build index / ask questions)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <this-repo-url>
cd RAG_Document_Question_Answering
pip install -r requirements.txt
```

This project uses a local LLM via Ollama for generation. Install it from [ollama.com](https://ollama.com), then pull a model once:

```bash
ollama pull llama3
```

## Usage

### Option 1: Jupyter Notebook (recommended for learning)

```bash
jupyter notebook notebook/RAG_Pipeline.ipynb
```

Walks through every pipeline stage interactively — ingestion, chunking, embeddings, retrieval, and generation.

### Option 2: Command Line

**Build the index from your own documents:**
```bash
python main.py --build --source folder --path data/raw
```

**Build the index from a Hugging Face dataset instead** (e.g. `squad`, `vectara/open_ragbench`):
```bash
python main.py --build --source hf --dataset squad --max_docs 200
```

**Ask a question:**
```bash
python main.py --ask "What is the main idea of the document?"
```

## Example

```
Question: What is the main idea of the document?

Answer: [Generated answer grounded in the retrieved document chunks]

Sources used:
 - notes.pdf::chunk_3
 - notes.pdf::chunk_7
```

## Improvements & Future Experiments

- **Better chunking strategies** — sentence-aware or semantic chunking instead of fixed-size splitting
- **Different embedding models** — e.g. `all-mpnet-base-v2` for higher-quality embeddings
- **Hybrid search** — combine keyword search (BM25) with vector similarity search
- **Re-ranking** — retrieve a larger candidate set, then re-rank with a cross-encoder for better relevance
- **Different LLMs** — swap the Ollama model, or plug in an API-based model instead

## Key Learnings

- How RAG systems combine retrieval and generation to ground LLM answers in real data
- Why retrieval quality directly determines answer quality
- Working hands-on with embeddings and vector databases
- Handling unstructured text data (PDFs, plain text)
- Designing a modular, scalable pipeline architecture

## License

This project is for educational purposes.
