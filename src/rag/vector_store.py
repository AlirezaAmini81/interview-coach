"""
Chroma-backed vector store. Was a hand-rolled Python list + numpy cosine
search (still true in spirit - this is a thin wrapper, not a different
architecture); switched to Chroma specifically so live-fetched content
(see rag/live_fetch.py) survives an app restart instead of being thrown
away every time.

Collections are created with cosine distance space (`hnsw:space: cosine`)
so scores stay comparable to the project's existing cosine-based
RELEVANCE_THRESHOLD. Chroma returns *distances*; `score = 1 - distance`
converts back to the same similarity scale used everywhere else - verified
against the old brute-force numpy cosine calculation before relying on it
(distance 0.0 -> similarity 1.0, distance 1.0 -> similarity 0.0, matches
exactly).
"""
from dataclasses import dataclass

import chromadb
import numpy as np

from .embeddings import Embedder


@dataclass
class Chunk:
    text: str
    source: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, collection_name: str, persist_path: str | None = None):
        """persist_path=None uses an in-memory client (what tests get by
        default - fresh and isolated per instance, no disk writes). A real
        path uses a client that persists to disk across process restarts."""
        self._client = chromadb.EphemeralClient() if persist_path is None else chromadb.PersistentClient(path=persist_path)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        start = self._collection.count()
        ids = [str(start + i) for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=np.asarray(vectors).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def chunks(self) -> list[Chunk]:
        result = self._collection.get()
        return [Chunk(text=doc, source=meta["source"]) for doc, meta in zip(result["documents"], result["metadatas"])]

    def texts(self) -> list[str]:
        return [c.text for c in self.chunks()]

    def replace(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Wholesale swap of contents - used when the embedder had to be
        refit (e.g. TfidfEmbedder after a live-fetched doc adds new
        vocabulary), so every existing vector needs recomputing too, not
        just the new ones appended.

        Drops and recreates the collection rather than just deleting its
        documents: Chroma locks a collection's embedding dimension to
        whatever was inserted first, but TfidfEmbedder's vector dimension
        changes every time its vocabulary changes (which is exactly when
        replace() gets called) - clearing documents alone would still
        leave the old, now-wrong dimension in place."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.add(chunks, vectors)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[np.asarray(query_vector).tolist()], n_results=k)
        chunks = [Chunk(text=doc, source=meta["source"]) for doc, meta in zip(result["documents"][0], result["metadatas"][0])]
        return [SearchResult(chunk=c, score=1.0 - dist) for c, dist in zip(chunks, result["distances"][0])]


def rebuild_store(store: VectorStore, chunks: list[Chunk], embedder: Embedder) -> None:
    """Fit the embedder on the full given chunk set and replace the
    store's contents wholesale. Shared by index.py (startup: local files +
    any previously live-fetched chunks recovered from persistence) and
    live_fetch.py (a single new fetch added to whatever's already there) -
    same operation either way, and it has to be the *full* set each time,
    not just the new chunks, because TfidfEmbedder's vectors are only
    meaningful relative to the vocabulary it was fit on."""
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed(texts)
    store.replace(chunks, np.asarray(vectors))
