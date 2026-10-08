"""Prompt templates and a helper to read the LLM response."""
from langchain_core.prompts import ChatPromptTemplate

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


def get_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))
