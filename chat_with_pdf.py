# ─────────────────────────────────────────────
#  RAG Chatbot — Streamlit App
# ─────────────────────────────────────────────

import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title = "Chat with PDF",
    page_icon  = "📄",
    layout     = "centered"
)

st.title("📄 Chat with your PDF")
st.caption("Powered by Gemini + LangChain + ChromaDB")

# ── SIDEBAR — CONFIG ──────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key  = st.text_input(
                    "Google API Key",
                    type        = "password",
                    placeholder = "Paste your key here..."
               )
    pdf_file = st.file_uploader(
                    "Upload your PDF",
                    type = ["pdf"]
               )
    st.divider()
    st.caption("Built with ❤️ using LangChain + Gemini")

# ── HELPER — FORMAT DOCS ──────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── LOAD & BUILD RAG ──────────────────────────
@st.cache_resource(show_spinner="🧠 Building vector store...")
def build_rag_chain(api_key, pdf_bytes, pdf_name):
    # Save uploaded PDF temporarily
    temp_path = f"./{pdf_name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    os.environ["GOOGLE_API_KEY"] = api_key

    # Step 1 — Load
    loader    = PyMuPDFLoader(temp_path)
    documents = loader.load()

    # Step 2 — Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1000,
        chunk_overlap = 200
    )
    chunks = splitter.split_documents(documents)

    # Step 3 — Embed + Store
    embeddings   = GoogleGenerativeAIEmbeddings(
                       model = "models/gemini-embedding-001"
                   )
    vector_store = Chroma.from_documents(
                       chunks,
                       embedding         = embeddings,
                       persist_directory = "./chroma_db"
                   )
    retriever    = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 4 — LLM
    llm = ChatGoogleGenerativeAI(
        model       = "gemini-2.5-flash",
        temperature = 0.2
    )

    # Step 5 — Prompt
    prompt = PromptTemplate.from_template("""
You are an expert assistant. Answer using ONLY the context below.
If the answer is not in the context, say:
"I couldn't find this in the document."

Context: {context}

Question: {question}

Answer:
""")

    # Step 6 — Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── MAIN APP ──────────────────────────────────
if not api_key:
    st.info("👈 Please enter your Google API Key in the sidebar!")
    st.stop()

if not pdf_file:
    st.info("👈 Please upload a PDF file in the sidebar!")
    st.stop()

# Build RAG chain
rag_chain = build_rag_chain(
    api_key   = api_key,
    pdf_bytes = pdf_file.read(),
    pdf_name  = pdf_file.name
)

st.success(f"✅ **{pdf_file.name}** loaded! Ask me anything about it.")
st.divider()

# ── CHAT HISTORY ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── CHAT INPUT ────────────────────────────────
if question := st.chat_input("Ask a question about your PDF..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            answer = rag_chain.invoke(question)
        st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})