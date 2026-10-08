"""Settings: paths, model names, chunk sizes and the API key."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GEMINI_API_KEY", None)
API_KEY = os.environ.get("GOOGLE_API_KEY")

PDF_PATH = ROOT / "data" / "documents" / "Retrieval-augmented_generation.pdf"
INDEX_DIR = ROOT / "vectorstore" / "faiss_index"

EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.8-flash"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
MAX_GRAPH_ATTEMPTS = 2

# Everything after this text in the PDF is the References section
REFERENCES_MARKER = '1. "What is retrieval-augmented generation?"'
