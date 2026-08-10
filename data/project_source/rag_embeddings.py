"""
Embedder abstraction: turns text into vectors so similarity search is possible.

TfidfEmbedder needs no model download and no internet access - good for a
fast, fully offline demo. It represents each chunk as a sparse "which words
appear how often" vector. It is NOT the same as a neural embedding model,
and you should be explicit about that distinction if this comes up in an
interview.

OllamaEmbedder is the real upgrade: dense neural embeddings from a local
model (e.g. nomic-embed-text), free and private, that capture meaning
rather than just word overlap. Use this whenever Ollama is available -
it's what "real RAG" actually means.
"""
import json
import urllib.request
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary/model from a corpus. Call once, before embed()."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of vectors."""
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() with the corpus before embed().")
        return self.vectorizer.transform(texts).toarray()


class OllamaEmbedder(Embedder):
    """Real neural embeddings via a local Ollama model - free, private, and
    genuinely semantic (captures meaning, not just word overlap, unlike
    TfidfEmbedder). Requires `ollama pull nomic-embed-text` (or pass a
    different model name you already have)."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def fit(self, texts: list[str]) -> None:
        pass  # pretrained model - nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._embed_one(text) for text in texts])

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url} for embeddings. "
                "Is it running? Is the model pulled?"
            ) from exc
        return data["embedding"]


class SentenceTransformerEmbedder(Embedder):
    """Alternative upgrade path if you'd rather not use Ollama for embeddings.
    Not used by default. Needs `pip install sentence-transformers` and
    internet access the first time it downloads the model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import: optional dependency

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        pass  # pretrained model, nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts)
