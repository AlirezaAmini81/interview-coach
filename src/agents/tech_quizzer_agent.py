"""
Generates one technical question grounded in the real docs corpus, given a
"quiz me on X" request - the mirror image of the Explainer. Returns both
the question and the exact excerpts used, so scoring later can be grounded
in the SAME material the question came from rather than re-retrieving and
risking a different (possibly less relevant) set of chunks.

Retrieval goes through live_fetch.retrieve_with_fetch, same as the
Explainer - if the requested topic isn't covered locally, it tries fetching
a real GitHub README before giving up. See rag/live_fetch.py.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.live_fetch import RELEVANCE_THRESHOLD, retrieve_with_fetch
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
    results = retrieve_with_fetch(topic_request, docs_store, docs_embedder, k=k)

    if not results or results[0].score < RELEVANCE_THRESHOLD:
        return (
            "No material found to quiz you on that topic - checked the local "
            "docs corpus and tried fetching real material for it, neither "
            "turned up anything usable.",
            "",
        )

    excerpts = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in results)
    user_message = f"Excerpts:\n\n{excerpts}\n\nRequest: {topic_request}"
    question = llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
    return question, excerpts
