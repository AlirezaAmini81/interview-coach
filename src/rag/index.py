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

project_store specifically can be built from MULTIPLE user projects (local
folders and/or GitHub repos), not just the one bundled sample - see
_build_project_store. Every project's chunks go into the SAME store, each
tagged with which project it came from via its source field
("project:{name}/{filename}" for local, "project:{name}:github:{path}" for
GitHub-fetched - see live_fetch.fetch_github_repo_files). There's
deliberately no separate "which project does this answer concern" step:
fact_checker_agent's search over the combined store naturally surfaces
whichever project's chunks are most similar to a given answer - the
retrieval itself IS the project-disambiguation mechanism.
"""
import os
import re

from .embeddings import Embedder, TfidfEmbedder
from .live_fetch import fetch_github_repo_files
from .vector_store import Chunk, INDEXABLE_EXTENSIONS, VectorStore, chunk_text, rebuild_store

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROJECT_SOURCE_DIR = os.path.join(DATA_DIR, "project_source")
TOOL_DOCS_DIR = os.path.join(DATA_DIR, "tool_docs")

DEFAULT_PROJECT_SOURCES = [{"name": "sample-project", "kind": "local", "location": PROJECT_SOURCE_DIR}]

_GITHUB_URL_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")
_OWNER_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def parse_project_location(text: str) -> tuple[str, str]:
    """'https://github.com/owner/repo' or bare 'owner/repo' -> ("github",
    "owner/repo"); anything else is treated as a local folder path. Used by
    streamlit_app.py's "Your projects" form - one text field, kind inferred
    from what's actually typed rather than a separate dropdown."""
    text = text.strip().rstrip("/")
    match = _GITHUB_URL_RE.search(text)
    if match:
        return "github", match.group(1)
    if _OWNER_REPO_RE.match(text):
        return "github", text
    return "local", text


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


def _read_local_project_chunks(name: str, directory: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(INDEXABLE_EXTENSIONS):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8") as f:
            chunks.extend(chunk_text(f.read(), source=f"project:{name}/{filename}"))
    return chunks


def _github_project_name(source: str) -> str | None:
    """Extract the project name back out of a 'project:{name}:github:{path}'
    source tag, or None if this isn't a GitHub-fetched project chunk."""
    if not source.startswith("project:") or ":github:" not in source:
        return None
    return source.removeprefix("project:").split(":github:", 1)[0]


def _build_project_store(
    project_sources: list[dict], embedder: Embedder, persist_path: str | None
) -> VectorStore:
    """Builds project_store from one or more user projects - local folders
    and/or GitHub repos, each tagged by name so a search over the combined
    store naturally surfaces the right project (see module docstring).

    Local projects are always re-read fresh (cheap, correctness). GitHub
    projects are only fetched once - recovered from persistence on later
    calls via their 'project:{name}:github:' tag, since fetching a repo's
    files is comparatively expensive (many API calls, see
    live_fetch.fetch_github_repo_files). A project removed from
    project_sources has its recovered chunks dropped here too, so removing
    it actually removes it rather than leaving orphaned content behind.
    """
    store = VectorStore("project_source", persist_path=persist_path)
    configured_names = {p["name"] for p in project_sources}

    recovered_github_chunks = [
        c for c in store.chunks() if _github_project_name(c.source) in configured_names
    ]
    already_fetched_names = {_github_project_name(c.source) for c in recovered_github_chunks}

    fresh_chunks: list[Chunk] = []
    for source in project_sources:
        name, kind, location = source["name"], source["kind"], source["location"]
        if kind == "local":
            fresh_chunks.extend(_read_local_project_chunks(name, location))
        elif kind == "github" and name not in already_fetched_names:
            fetched = fetch_github_repo_files(location)
            fresh_chunks.extend(Chunk(text=c.text, source=f"project:{name}:github:{c.source}") for c in fetched)

    rebuild_store(store, fresh_chunks + recovered_github_chunks, embedder)
    return store


def build_indices(
    embedder_factory=TfidfEmbedder, persist_path: str | None = None, project_sources: list[dict] | None = None
) -> tuple[VectorStore, Embedder, VectorStore, Embedder]:
    """Returns (project_store, project_embedder, docs_store, docs_embedder).

    Two separate embedders too, not just two stores - TfidfEmbedder's
    vocabulary is fit per-corpus, and the code corpus and the concept-docs
    corpus have very different vocabularies (Python syntax vs prose).

    persist_path=None (the default, and what tests use) keeps everything
    in-memory only, matching the old always-rebuild-from-scratch behavior.
    Pass a real directory path to persist across restarts - see
    streamlit_app.py / main.py.

    project_sources=None defaults to DEFAULT_PROJECT_SOURCES (the single
    bundled sample project) - fully backward compatible. Pass a list of
    {"name", "kind": "local"|"github", "location"} dicts to check answers
    against real user projects instead - see streamlit_app.py's "Your
    projects" sidebar section and _build_project_store's docstring.
    """
    if project_sources is None:
        project_sources = DEFAULT_PROJECT_SOURCES
    project_embedder = embedder_factory()
    project_store = _build_project_store(project_sources, project_embedder, persist_path)

    docs_embedder = embedder_factory()
    docs_store = _build_index_from_dir(TOOL_DOCS_DIR, "tool_docs", docs_embedder, persist_path)

    return project_store, project_embedder, docs_store, docs_embedder
