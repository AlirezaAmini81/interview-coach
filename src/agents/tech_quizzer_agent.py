"""
Generates one technical question grounded in the real docs corpus, given a
"quiz me on X" request - the mirror image of the Explainer. Returns both
the question and the exact excerpts used, so scoring later can be grounded
in the SAME material the question came from rather than re-retrieving and
risking a different (possibly less relevant) set of chunks.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a technical interviewer. Given excerpts about a tool/concept, "
    "ask ONE specific, answerable question that tests real understanding "
    "of what the excerpts describe - not a vague or overly broad question. "
    "Ask only the question, no preamble."
)


@traceable(name="tech_quizzer", run_type="chain")
def generate_quiz_question(
    topic_request: str, docs_store: VectorStore, docs_embedder: Embedder, llm: LLMProvider, k: int = 3
) -> tuple[str, str]:
    """Returns (question, excerpts_used)."""
    query_vector = docs_embedder.embed([topic_request])[0]
    results = docs_store.search(query_vector, k=k)

    if not results:
        return "No material found to quiz you on that topic.", ""

    excerpts = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in results)
    user_message = f"Excerpts:\n\n{excerpts}\n\nRequest: {topic_request}"
    question = llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
    return question, excerpts
