"""
A deliberately minimal vector store: a list of vectors plus cosine similarity
search. Real projects use Chroma/Pinecone/FAISS because they scale and
persist to disk - but for a few dozen chunks, hand-rolling this makes the
actual mechanism visible instead of hidden inside a library call. Good for
understanding it, easy to swap out later if you need it to scale.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    text: str
    source: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        similarities = _cosine_similarity(query_vector, self._vectors)
        top_k_idx = np.argsort(similarities)[::-1][:k]
        return [SearchResult(chunk=self._chunks[i], score=float(similarities[i])) for i in top_k_idx]


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query) + 1e-10
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-10
    return (matrix @ query) / (matrix_norms * query_norm)
