"""
Builds two separate vector indices, because the two RAG uses in this
project are genuinely different:

- project_index: your real project's README + source code. Used by the
  Fact-Checker to verify self-description answers against ground truth.
- docs_index: original explainer docs on LangGraph/LangChain/LangSmith/RAG/
  multi-agent concepts. Used by the Explainer for grounded answers and
  comparisons.

Keeping them separate matters: mixing them would let a fact-check query
accidentally retrieve a generic concept explanation instead of your actual
code, or vice versa.
"""
import os

from .embeddings import Embedder, TfidfEmbedder
from .vector_store import Chunk, VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROJECT_SOURCE_DIR = os.path.join(DATA_DIR, "project_source")
TOOL_DOCS_DIR = os.path.join(DATA_DIR, "tool_docs")

INDEXABLE_EXTENSIONS = (".txt", ".md", ".py")


def _chunk_text(text: str, source: str, max_chars: int = 600) -> list[Chunk]:
    """Split on blank lines; merge short paragraphs up to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buffer = [], ""
    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) > max_chars:
            chunks.append(Chunk(text=buffer.strip(), source=source))
            buffer = ""
        buffer += ("\n\n" if buffer else "") + paragraph
    if buffer:
        chunks.append(Chunk(text=buffer.strip(), source=source))
    return chunks


def _build_index_from_dir(directory: str, embedder: Embedder) -> VectorStore:
    all_chunks: list[Chunk] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(INDEXABLE_EXTENSIONS):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:
            all_chunks.extend(_chunk_text(f.read(), source=filename))

    embedder.fit([c.text for c in all_chunks])
    vectors = embedder.embed([c.text for c in all_chunks])

    store = VectorStore()
    store.add(all_chunks, vectors)
    return store


def build_indices(embedder_factory=TfidfEmbedder) -> tuple[VectorStore, Embedder, VectorStore, Embedder]:
    """Returns (project_store, project_embedder, docs_store, docs_embedder).

    Two separate embedders too, not just two stores - TfidfEmbedder's
    vocabulary is fit per-corpus, and the code corpus and the concept-docs
    corpus have very different vocabularies (Python syntax vs prose).
    """
    project_embedder = embedder_factory()
    project_store = _build_index_from_dir(PROJECT_SOURCE_DIR, project_embedder)

    docs_embedder = embedder_factory()
    docs_store = _build_index_from_dir(TOOL_DOCS_DIR, docs_embedder)

    return project_store, project_embedder, docs_store, docs_embedder
