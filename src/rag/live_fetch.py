"""
On-demand corpus augmentation: when a question touches a tool/concept the
local `tool_docs` corpus doesn't cover, fetch a real README from GitHub
rather than either hallucinating an answer or just refusing outright.

RELEVANCE_THRESHOLD was picked empirically against the current tool_docs
corpus: bare topic names for in-corpus tools (LangGraph, LangChain,
LangSmith, RAG) scored 0.24-0.45 cosine similarity, while genuinely
out-of-corpus tools (CrewAI, DSPy, Model Context Protocol) scored 0.0-0.19.
Retune if the corpus changes significantly. Deliberately checked against
BARE topic names, not full questions - question scaffolding ("explain",
"quiz me on") pollutes TF-IDF similarity enough to blur that separation
(see extract_topics).

Uses urllib only (no new dependency), same pattern as
OllamaEmbedder._embed_one in embeddings.py. Any network failure - offline,
GitHub rate-limited, nothing found - must degrade to returning None, never
raise. Callers fall back to an honest "nothing found" response rather than
crash or silently answer ungrounded. Note GitHub's *search* endpoint has
its own much tighter unauthenticated limit (10 req/min) than the general
API (60 req/hour) - each fetch can use up to 2 search calls (see
_search_repo's in:name-first-then-fallback), so a burst of several
back-to-back explain/quiz requests for uncovered topics can hit it. When
that happens the request comes back as a normal HTTP error, which is
already handled the same as any other failure - it just means "nothing
found" is sometimes actually "try again in a few seconds," indistinguishable
in the current response text. Not fixed here (would need surfacing a
distinct message for that specific case) - a known, minor honesty gap.
"""
import json
import re
import urllib.parse
import urllib.request

from .embeddings import Embedder
from .index import chunk_text
from .vector_store import Chunk, SearchResult, VectorStore, rebuild_store

RELEVANCE_THRESHOLD = 0.2
GITHUB_API = "https://api.github.com"
MAX_README_CHARS = 6000
REQUEST_TIMEOUT = 6

# Stripped from the front/end of a question before treating the remainder
# as a topic/entity name - order matters, longer/more specific phrases
# first. This is a regex heuristic, not real NLP - it covers the common
# phrasings this project's own prompts and examples use, not every possible
# way to ask a question.
_LEADING_PHRASES = [
    "what is the difference between", "what's the difference between", "difference between",
    "what are", "what is", "what's", "how does", "how do",
    "quiz me on", "test me on", "ask me something about", "ask me about",
    "tell me about", "explain", "compare",
]
_TRAILING_PHRASES = ["to me", "for me", "please", "in detail"]
_CONNECTORS = re.compile(r"\s+(?:vs\.?|versus|and)\s+", re.IGNORECASE)


def _strip_phrase(text: str, phrases: list[str], from_start: bool) -> str:
    lowered = text.lower()
    for phrase in phrases:
        if from_start and lowered.startswith(phrase):
            return text[len(phrase):].strip()
        if not from_start and lowered.endswith(phrase):
            return text[: len(text) - len(phrase)].strip()
    return text


def extract_topics(question: str) -> list[str]:
    """Strip question scaffolding, then split comparison-style questions
    into their separate topics. 'compare LangGraph and CrewAI' -> ['LangGraph',
    'CrewAI']. 'explain docker to me' -> ['docker'] (leading AND trailing
    filler both stripped - 'to me' left dangling would otherwise become part
    of the search query)."""
    text = question.strip().rstrip("?").strip()
    text = _strip_phrase(text, _LEADING_PHRASES, from_start=True)
    text = _strip_phrase(text, _TRAILING_PHRASES, from_start=False)

    parts = _CONNECTORS.split(text)
    return [p.strip() for p in parts if p.strip()]


def _github_get(url: str, headers: dict) -> bytes | None:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read()
    except OSError:
        return None


def _search_repo(topic: str, restrict_to_name: bool) -> dict | None:
    query = urllib.parse.quote(f"{topic} in:name" if restrict_to_name else topic)
    search_url = f"{GITHUB_API}/search/repositories?q={query}&sort=stars&order=desc&per_page=1"
    search_body = _github_get(search_url, headers={"Accept": "application/vnd.github+json"})
    if search_body is None:
        return None
    try:
        items = json.loads(search_body).get("items", [])
    except json.JSONDecodeError:
        return None
    return items[0] if items else None


def fetch_github_readme(topic: str) -> Chunk | None:
    """Find a GitHub repo matching `topic` and return its README as a
    single Chunk, or None if nothing was found or the network call failed.

    Searches with `in:name` first (repo name must contain the topic) - an
    unqualified search ranks purely by star count among anything that
    mentions the topic ANYWHERE (readme, topics, description), which can
    surface a hugely popular but unrelated repo instead of the tool itself
    (e.g. "docker" unqualified returned oh-my-zsh, which just happens to
    mention Docker as one of its many plugins, ahead of docker/compose).
    Falls back to an unqualified search only if the name-restricted one
    finds nothing - needed for multi-word concepts that aren't literally a
    repo name (e.g. "groundedness").
    """
    item = _search_repo(topic, restrict_to_name=True) or _search_repo(topic, restrict_to_name=False)
    if item is None:
        return None

    full_name = item.get("full_name")
    if not full_name:
        return None

    readme_url = f"{GITHUB_API}/repos/{full_name}/readme"
    readme_body = _github_get(readme_url, headers={"Accept": "application/vnd.github.raw+json"})
    if not readme_body:
        return None

    text = readme_body.decode("utf-8", errors="ignore")[:MAX_README_CHARS]
    if not text.strip():
        return None
    return Chunk(text=text, source=f"github:{full_name}/README.md")


def augment_with_live_fetch(topic: str, store: VectorStore, embedder: Embedder) -> bool:
    """Fetch real material for `topic`, chunk it, and merge it into `store`
    via rebuild_store (refits the embedder on the combined old+new text and
    re-embeds everything, not just the new chunks - required for
    TfidfEmbedder, since .transform() silently drops terms outside the
    previously-fit vocabulary, which would otherwise make a freshly
    fetched doc score near-zero for its own defining terms). Cheap here
    since the corpus is a few dozen chunks. If `store` was built with a
    persist_path (see index.py), this survives app restarts - recovered
    automatically next time build_indices() runs via its "github:" source
    prefix check. Returns True if anything was actually added."""
    fetched_chunk = fetch_github_readme(topic)
    if fetched_chunk is None:
        return False

    new_chunks = chunk_text(fetched_chunk.text, source=fetched_chunk.source)
    if not new_chunks:
        return False

    rebuild_store(store, store.chunks() + new_chunks, embedder)
    return True


def retrieve_with_fetch(query: str, store: VectorStore, embedder: Embedder, k: int = 3) -> list[SearchResult]:
    """The retrieval entry point for explainer/quizzer: search locally
    first, and for any extracted topic that's below the relevance
    threshold locally, try to fetch real material for it before giving up.
    Always returns whatever `store.search` returns after that - possibly
    still weak results if fetching failed too, so callers should still
    check relevance themselves before treating the result as grounded."""
    query_vector = embedder.embed([query])[0]
    results = store.search(query_vector, k=k)

    fetched_anything = False
    for topic in extract_topics(query):
        topic_vector = embedder.embed([topic])[0]
        topic_results = store.search(topic_vector, k=1)
        best_score = topic_results[0].score if topic_results else 0.0
        if best_score < RELEVANCE_THRESHOLD and augment_with_live_fetch(topic, store, embedder):
            fetched_anything = True

    if fetched_anything:
        # Vocabulary changed (TfidfEmbedder was refit), so the original
        # query's own vector is stale - re-embed before the real search.
        query_vector = embedder.embed([query])[0]
        results = store.search(query_vector, k=k)

    return results
