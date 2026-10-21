import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Another installed package may already have claimed the name "src".
# Remove it so our own src folder is loaded.
for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
    del sys.modules[name]

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.graph import build_graph, run
from src.retriever import get_llm, load_db, search

USE_GRADING = False  # False = 1 Gemini call per question (saves quota). True = generate + grade.

# Gemini free tier allows about 100 embedding requests per minute (1 chunk = 1 request).
# So we embed in batches and wait between them. Bigger chunks = fewer chunks = fewer requests.
UPLOAD_CHUNK_SIZE = 1500
UPLOAD_CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 80      # stay safely under the 100 per minute limit
EMBED_WAIT_SECONDS = 65    # wait between batches so the per-minute quota resets
MAX_RETRIES = 3


# ---------- Default document (built with: python -m src.ingest) ----------
@st.cache_resource
def load_resources():
    db = load_db()
    llm = get_llm()
    return db, build_graph(db, llm, use_grading=USE_GRADING)


def get_embeddings():
    """Reuse the embedding model of the default index. Returns None if there is no default index."""
    if not config.INDEX_DIR.exists():
        return None
    base_db, _ = load_resources()
    return getattr(base_db, "embeddings", None) or base_db.embedding_function


# ---------- Upload: files -> chunks -> new vector store ----------
def load_uploaded_file(uploaded_file):
    """Save the upload to a temp file, read it with a loader, then delete the temp file."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        else:
            docs = TextLoader(tmp_path, encoding="utf-8").load()
    finally:
        os.remove(tmp_path)
    for d in docs:
        d.metadata["source"] = uploaded_file.name  # real file name instead of the temp path
    return docs


def split_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        docs.extend(load_uploaded_file(f))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=UPLOAD_CHUNK_SIZE, chunk_overlap=UPLOAD_CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def is_rate_limit_error(err):
    text = str(err)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def wait_with_countdown(seconds, message):
    box = st.empty()
    for remaining in range(seconds, 0, -1):
        box.info(f"{message} Resuming in {remaining}s...")
        time.sleep(1)
    box.empty()


def embed_in_batches(chunks, embeddings):
    """Build a FAISS index batch by batch, pausing between batches to respect the rate limit."""
    total = len(chunks)
    progress = st.progress(0.0, text=f"Embedding 0 / {total} chunks")
    db = None
    done = 0

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                if db is None:
                    db = FAISS.from_documents(batch, embeddings)
                else:
                    db.add_documents(batch)
                break
            except Exception as err:
                if is_rate_limit_error(err) and attempt < MAX_RETRIES - 1:
                    wait_with_countdown(EMBED_WAIT_SECONDS, "Rate limit reached.")
                else:
                    raise

        done += len(batch)
        progress.progress(done / total, text=f"Embedding {done} / {total} chunks")

        if done < total:
            wait_with_countdown(EMBED_WAIT_SECONDS, "Free-tier limit: pausing between batches.")

    progress.empty()
    return db


def get_active():
    """Use the uploaded index if there is one, otherwise the default index."""
    if "db" in st.session_state:
        return st.session_state.db, st.session_state.rag_app
    return load_resources()


def ask(question):
    """Returns (answer, source_chunks, used_fallback)."""
    db, rag_app = get_active()
    try:
        result = run(rag_app, question)
        return result["answer"], [d.page_content for d in result["docs"]], False
    except Exception:
        # Gemini busy or quota finished: show the most relevant passages instead
        sources = [d.page_content for d in search(db, question)]
        answer = (
            "The AI service is busy or its quota is finished right now, "
            "so here are the most relevant passages from the document:\n\n"
            + "\n\n---\n\n".join(sources)
        )
        return answer, sources, True


# ---------- Page ----------
st.title("RAG Chatbot")

if not config.API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Index documents", type="primary", disabled=not uploaded_files):
        embeddings = get_embeddings()
        if embeddings is None:
            st.error(
                "No default index found, so the embedding model could not be loaded. "
                "Run `python -m src.ingest` once first."
            )
        else:
            with st.spinner("Reading your documents..."):
                chunks = split_uploads(uploaded_files)

            if not chunks:
                st.error("No text could be extracted. The PDF may be a scanned image (OCR is needed).")
            else:
                n_batches = -(-len(chunks) // EMBED_BATCH_SIZE)  # ceiling division
                if n_batches > 1:
                    est_min = (n_batches - 1) * EMBED_WAIT_SECONDS // 60 + 1
                    st.warning(
                        f"{len(chunks)} chunks found. Free-tier limits mean this will take "
                        f"about {est_min} minute(s). Please keep this page open."
                    )
                try:
                    new_db = embed_in_batches(chunks, embeddings)
                except Exception as err:
                    if is_rate_limit_error(err):
                        st.error(
                            "Gemini embedding quota is exhausted. Wait a minute and try again, "
                            "upload a smaller file, or enable billing for your API key."
                        )
                    else:
                        st.error(f"Indexing failed: {err}")
                else:
                    st.session_state.db = new_db
                    st.session_state.rag_app = build_graph(new_db, get_llm(), use_grading=USE_GRADING)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")

    if "db" in st.session_state and st.button("Use default document"):
        del st.session_state["db"]
        del st.session_state["rag_app"]
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

using_upload = "db" in st.session_state
st.caption(
    "Asking about: your uploaded documents."
    if using_upload
    else "Asking about: the default Retrieval-augmented generation document."
)

if not using_upload and not config.INDEX_DIR.exists():
    st.error("Saved index not found. Run:  python -m src.ingest")
    st.stop()

# ---------- Chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources") and not m.get("fallback"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(s[:400] + "...")
                    st.divider()

if question := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, fallback = ask(question)
        st.write(answer)
        if sources and not fallback:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s[:400] + "...")
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "fallback": fallback}
    )