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

Each store recovers any previously live-fetched content (see
rag/live_fetch.py) from persistence before rebuilding - "github:" is the
source prefix that marks a chunk as fetched rather than a local file (see
fetch_github_readme). Local files are always re-read fresh on every
startup (cheap, and guarantees correctness if they changed); only the
fetched half of the corpus needs recovering.
"""
import os

from .embeddings import Embedder, TfidfEmbedder
from .vector_store import Chunk, VectorStore, rebuild_store

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROJECT_SOURCE_DIR = os.path.join(DATA_DIR, "project_source")
TOOL_DOCS_DIR = os.path.join(DATA_DIR, "tool_docs")

INDEXABLE_EXTENSIONS = (".txt", ".md", ".py")


def chunk_text(text: str, source: str, max_chars: int = 600) -> list[Chunk]:
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


def _read_local_chunks(directory: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(INDEXABLE_EXTENSIONS):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:
            all_chunks.extend(chunk_text(f.read(), source=filename))
    return all_chunks


def _build_index_from_dir(directory: str, collection_name: str, embedder: Embedder, persist_path: str | None) -> VectorStore:
    store = VectorStore(collection_name, persist_path=persist_path)

    # Recover anything live-fetched last session (empty on first-ever run -
    # store.chunks() on a fresh collection is just []). Local files are
    # always read fresh below regardless, so only the fetched half needs
    # recovering here.
    fetched_chunks = [c for c in store.chunks() if c.source.startswith("github:")]
    local_chunks = _read_local_chunks(directory)

    rebuild_store(store, local_chunks + fetched_chunks, embedder)
    return store


def build_indices(embedder_factory=TfidfEmbedder, persist_path: str | None = None) -> tuple[VectorStore, Embedder, VectorStore, Embedder]:
    """Returns (project_store, project_embedder, docs_store, docs_embedder).

    Two separate embedders too, not just two stores - TfidfEmbedder's
    vocabulary is fit per-corpus, and the code corpus and the concept-docs
    corpus have very different vocabularies (Python syntax vs prose).

    persist_path=None (the default, and what tests use) keeps everything
    in-memory only, matching the old always-rebuild-from-scratch behavior.
    Pass a real directory path to persist across restarts - see
    streamlit_app.py / main.py.
    """
    project_embedder = embedder_factory()
    project_store = _build_index_from_dir(PROJECT_SOURCE_DIR, "project_source", project_embedder, persist_path)

    docs_embedder = embedder_factory()
    docs_store = _build_index_from_dir(TOOL_DOCS_DIR, "tool_docs", docs_embedder, persist_path)

    return project_store, project_embedder, docs_store, docs_embedder
