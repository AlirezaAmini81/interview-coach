"""
Checks a self-description answer against the REAL project source, not
memory or plausibility. This is the agent that catches "I used LangChain's
retriever" when you actually hand-built one - the whole reason RAG is
necessary here rather than decorative.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..llm.schemas import FactCheckVerdict
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You verify a candidate's description of their own project against "
    "real source excerpts from that project. If the answer doesn't "
    "describe implementation details at all, verdict is not_applicable. "
    "Otherwise, check every implementation claim against the excerpts."
)


@traceable(name="fact_checker", run_type="chain")
def check_answer(
    answer: str, project_store: VectorStore, project_embedder: Embedder, llm: LLMProvider, k: int = 3
) -> dict:
    query_vector = project_embedder.embed([answer])[0]
    results = project_store.search(query_vector, k=k)
    excerpts = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in results)

    user_message = f"Real project excerpts:\n\n{excerpts}\n\nCandidate's answer:\n{answer}"
    try:
        return llm.complete_json([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT, schema=FactCheckVerdict)
    except (ValueError, KeyError):
        return {"verdict": "not_applicable", "reasoning": "fact-check call failed"}
