"""Models and vector store loading (retrieval side)."""
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from . import config


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL, api_key=config.API_KEY)


def get_llm():
    return ChatGoogleGenerativeAI(model=config.LLM_MODEL, api_key=config.API_KEY, max_retries=2)


def load_db():
    """Load the saved FAISS index from disk."""
    return FAISS.load_local(
        str(config.INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def search(db, question, k=config.TOP_K):
    """Top-k most similar chunks (no LLM call)."""
    return db.similarity_search(question, k=k)
