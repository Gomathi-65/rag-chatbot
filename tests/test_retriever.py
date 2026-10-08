"""Basic tests for the retrieval part of the RAG pipeline.

Run from the project root:
    python -m pytest tests -q

The first three tests need no API key and make no Gemini calls.
The last test uses the saved FAISS index and embeds the question
(embedding quota only, no LLM call). It is skipped if the index or key is missing.
"""
import pytest

from src import config
from src.ingest import clean_text, split_text


def test_clean_text_removes_references():
    text = "Main content about RAG. " + config.REFERENCES_MARKER + " IBM link, doi numbers"
    cleaned = clean_text(text)
    assert "Main content about RAG." in cleaned
    assert "IBM link" not in cleaned


def test_clean_text_keeps_text_without_references():
    text = "A short text with no references section."
    assert clean_text(text) == text


def test_split_text_respects_chunk_size():
    text = "RAG retrieves relevant passages before generating an answer. " * 100
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= config.CHUNK_SIZE for c in chunks)


@pytest.mark.skipif(
    not config.INDEX_DIR.exists() or not config.API_KEY,
    reason="needs the saved index and GOOGLE_API_KEY",
)
def test_retrieval_returns_relevant_chunks():
    from src.retriever import load_db, search

    db = load_db()
    results = search(db, "What is RAG?", k=3)
    assert len(results) == 3
    joined = " ".join(r.page_content.lower() for r in results)
    assert "retrieval" in joined