"""LangGraph state graph: retrieve -> generate -> grade -> (retry)."""
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from . import config
from .chains import answer_prompt, get_text, grade_prompt


class RAGState(TypedDict):
    question: str
    docs: List
    answer: str
    grounded: bool
    retries: int


def build_graph(db, llm, use_grading=False):
    def retrieve(state: RAGState):
        k = config.TOP_K + 2 * state["retries"]
        return {"docs": db.similarity_search(state["question"], k=k)}

    def generate(state: RAGState):
        context = "\n\n".join(d.page_content for d in state["docs"])
        msg = answer_prompt.invoke({"context": context, "question": state["question"]})
        return {"answer": get_text(llm.invoke(msg))}

    def grade(state: RAGState):
        if not use_grading:
            return {"grounded": True, "retries": state["retries"] + 1}
        try:
            context = "\n\n".join(d.page_content for d in state["docs"])
            msg = grade_prompt.invoke({"context": context, "answer": state["answer"]})
            verdict = get_text(llm.invoke(msg)).strip().lower()
            return {"grounded": verdict.startswith("yes"), "retries": state["retries"] + 1}
        except Exception:
            # If grading fails, keep the answer we already have
            return {"grounded": True, "retries": state["retries"] + 1}

    def decide(state: RAGState):
        if state["grounded"] or state["retries"] >= config.MAX_GRAPH_ATTEMPTS:
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


def run(rag_app, question):
    return rag_app.invoke(
        {"question": question, "docs": [], "answer": "", "grounded": False, "retries": 0}
    )