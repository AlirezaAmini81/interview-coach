"""
Answers the user's own questions about tools/concepts, grounded in the real
docs corpus. Detects comparison-style questions and retrieves more broadly
so both sides of a comparison have real source material, rather than only
ever answering about one topic at a time.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a technical explainer. Answer ONLY using the provided "
    "excerpts - if they don't cover something, say so rather than filling "
    "gaps from outside knowledge. If the question asks for a comparison, "
    "structure your answer around the actual tradeoffs found in the "
    "excerpts, not a generic comparison. Cite the source file in brackets."
)

COMPARISON_SIGNALS = ["compare", " vs ", " vs. ", "versus", "difference between"]


@traceable(name="explainer", run_type="chain")
def explain(question: str, docs_store: VectorStore, docs_embedder: Embedder, llm: LLMProvider) -> str:
    is_comparison = any(signal in question.lower() for signal in COMPARISON_SIGNALS)
    k = 5 if is_comparison else 3  # wider net so both sides of a comparison get real material

    query_vector = docs_embedder.embed([question])[0]
    results = docs_store.search(query_vector, k=k)

    if not results:
        return "No relevant material found in the docs corpus for that question."

    excerpts = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in results)
    user_message = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
