"""
The one agent that does RAG. Everything else in the system routes to this
correctly - this is where retrieval actually happens.
"""
from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY using the provided excerpts. "
    "If the excerpts don't contain the answer, say so - do not use outside "
    "knowledge. Cite the source file in brackets after each claim."
)


def run(
    instruction: str,
    vector_store: VectorStore,
    embedder: Embedder,
    llm: LLMProvider,
    k: int = 3,
) -> str:
    query_vector = embedder.embed([instruction])[0]
    results = vector_store.search(query_vector, k=k)

    if not results:
        return "No relevant excerpts found in the loaded papers."

    context = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in results)
    user_message = f"Excerpts:\n\n{context}\n\nQuestion: {instruction}"

    return llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
