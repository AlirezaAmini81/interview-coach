"""
The whole point of having two separate indices is that they don't bleed
into each other - a question about your code shouldn't retrieve concept
docs, and vice versa. These tests check exactly that.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.embeddings import TfidfEmbedder
from src.rag.index import build_indices


def test_project_index_finds_real_code_not_concept_docs():
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)

    query = "How does the vector store search for similar chunks?"
    query_vector = project_embedder.embed([query])[0]
    results = project_store.search(query_vector, k=1)

    assert len(results) == 1
    assert results[0].chunk.source == "rag_vector_store.py"


def test_docs_index_finds_concept_docs_not_project_code():
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)

    query = "What is groundedness in RAG?"
    query_vector = docs_embedder.embed([query])[0]
    results = docs_store.search(query_vector, k=1)

    assert len(results) == 1
    assert results[0].chunk.source == "rag_concepts.txt"


def test_indices_are_genuinely_separate():
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)

    # A code-specific query should not surface a concept-doc source, and
    # vice versa - confirms the two corpora aren't accidentally merged.
    project_sources = {c.source for c in project_store._chunks}
    docs_sources = {c.source for c in docs_store._chunks}
    assert project_sources.isdisjoint(docs_sources)
    assert "rag_vector_store.py" in project_sources
    assert "rag_concepts.txt" in docs_sources
