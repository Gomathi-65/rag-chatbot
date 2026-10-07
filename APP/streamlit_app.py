import os
from pathlib import Path
from typing import TypedDict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ---------- Settings ----------
ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = str(ROOT / "vectorstore" / "faiss_index")
USE_GRADING = False  # False = 1 Gemini call per question (saves quota). True = generate + grade.

load_dotenv(ROOT / ".env", override=True)
os.environ.pop("GEMINI_API_KEY", None)
API_KEY = os.environ.get("GOOGLE_API_KEY")


# ---------- Load models and index (once) ----------
@st.cache_resource
def load_resources():
    embed = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", api_key=API_KEY, max_retries=2)
    db = FAISS.load_local(INDEX_DIR, embed, allow_dangerous_deserialization=True)
    return db, llm


def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ---------- LangGraph ----------
class RAGState(TypedDict):
    question: str
    docs: List
    answer: str
    grounded: bool
    retries: int


answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""
)

grade_prompt = ChatPromptTemplate.from_template(
    """Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context? Reply with only "yes" or "no"."""
)


@st.cache_resource
def build_graph():
    db, llm = load_resources()

    def retrieve(state: RAGState):
        k = 3 + 2 * state["retries"]
        return {"docs": db.similarity_search(state["question"], k=k)}

    def generate(state: RAGState):
        context = "\n\n".join(d.page_content for d in state["docs"])
        msg = answer_prompt.invoke({"context": context, "question": state["question"]})
        return {"answer": get_text(llm.invoke(msg))}

    def grade(state: RAGState):
        if not USE_GRADING:
            return {"grounded": True, "retries": state["retries"] + 1}
        try:
            context = "\n\n".join(d.page_content for d in state["docs"])
            msg = grade_prompt.invoke({"context": context, "answer": state["answer"]})
            verdict = get_text(llm.invoke(msg)).strip().lower()
            return {"grounded": verdict.startswith("yes"), "retries": state["retries"] + 1}
        except Exception:
            return {"grounded": True, "retries": state["retries"] + 1}

    def decide(state: RAGState):
        if state["grounded"] or state["retries"] >= 2:
            return "end"
        return "retry"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", decide, {"retry": "retrieve", "end": END})
    return graph.compile()


def ask(question):
    """Returns (answer, source_chunks, used_fallback)."""
    db, _ = load_resources()
    rag_app = build_graph()
    try:
        result = rag_app.invoke(
            {"question": question, "docs": [], "answer": "", "grounded": False, "retries": 0}
        )
        sources = [d.page_content for d in result["docs"]]
        return result["answer"], sources, False
    except Exception as e:
        # Gemini busy or quota finished: show the most relevant passages instead
        docs = db.similarity_search(question, k=3)
        sources = [d.page_content for d in docs]
        answer = (
            "The AI service is busy or its quota is finished right now, "
            "so here are the most relevant passages from the document:\n\n"
            + "\n\n---\n\n".join(sources)
        )
        answer += f"\n\n(Debug: {type(e).__name__}: {str(e)[:200]})"
        return answer, sources, True


# ---------- UI ----------
st.title("RAG Chatbot")
st.caption("Ask questions about the Retrieval-augmented generation document.")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()

if not Path(INDEX_DIR).exists():
    st.error("Saved index not found at vectorstore/faiss_index. Run the notebook (Cell 7) first.")
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