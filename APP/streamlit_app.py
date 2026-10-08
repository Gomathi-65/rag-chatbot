import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Another installed package may already have claimed the name "src".
# Remove it so our own src folder is loaded.
for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
    del sys.modules[name]

import streamlit as st

from src import config
from src.graph import build_graph, run
from src.retriever import get_llm, load_db, search

USE_GRADING = False  # False = 1 Gemini call per question (saves quota). True = generate + grade.


@st.cache_resource
def load_resources():
    db = load_db()
    llm = get_llm()
    return db, build_graph(db, llm, use_grading=USE_GRADING)


def ask(question):
    """Returns (answer, source_chunks, used_fallback)."""
    db, rag_app = load_resources()
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


st.title("RAG Chatbot")
st.caption("Ask questions about the Retrieval-augmented generation document.")

if not config.API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()

if not config.INDEX_DIR.exists():
    st.error("Saved index not found. Run:  python -m src.ingest")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

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

