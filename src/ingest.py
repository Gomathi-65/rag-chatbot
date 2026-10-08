"""Load the PDF, clean it, split into chunks, embed and save the FAISS index.

Run once from the project root:
    python -m src.ingest
"""
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .retriever import get_embeddings


def load_text(pdf_path=config.PDF_PATH):
    reader = PdfReader(str(pdf_path))
    return "".join((page.extract_text() or "") for page in reader.pages)


def clean_text(text):
    cut = text.find(config.REFERENCES_MARKER)
    return text[:cut] if cut != -1 else text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def build_index():
    text = load_text()
    cleaned = clean_text(text)
    chunks = split_text(cleaned)
    print(f"Text: {len(text)} -> {len(cleaned)} characters, {len(chunks)} chunks")

    db = FAISS.from_texts(chunks, embedding=get_embeddings())
    config.INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(config.INDEX_DIR))
    print(f"Saved index to {config.INDEX_DIR}")
    return db


if __name__ == "__main__":
    build_index()
