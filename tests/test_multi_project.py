"""
Multiple user projects share ONE project_store, each chunk tagged by
project name - a search over an answer's text should naturally surface
whichever project's content is actually relevant, without any separate
"which project" selection step. See index.py's module docstring and
_build_project_store.
"""
import gc
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.embeddings import TfidfEmbedder
from src.rag.index import build_indices, parse_project_location
from src.rag.vector_store import Chunk


class _TempPersistDir:
    """Like tempfile.TemporaryDirectory, but tolerates cleanup failure -
    chromadb's PersistentClient keeps a file handle open on its index file
    (data_level0.bin) that Windows won't let get deleted until the handle
    is released, which doesn't reliably happen by the time the context
    exits. Not a bug in the code under test, just Windows file-locking
    semantics differing from POSIX (which allows deleting open files)."""

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, *exc_info):
        gc.collect()  # best-effort: encourage chroma's client to release its handle
        shutil.rmtree(self.path, ignore_errors=True)


def test_parse_project_location_detects_github_url():
    assert parse_project_location("https://github.com/someuser/some-repo") == ("github", "someuser/some-repo")


def test_parse_project_location_detects_owner_repo_shorthand():
    assert parse_project_location("someuser/some-repo") == ("github", "someuser/some-repo")


def test_parse_project_location_treats_other_text_as_local_path():
    assert parse_project_location(r"D:\projects\my-app") == ("local", r"D:\projects\my-app")


def test_two_local_projects_are_both_indexed_and_correctly_disambiguated():
    with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
        with open(os.path.join(dir_a, "main.py"), "w", encoding="utf-8") as f:
            f.write("def scrape_product_prices(url):\n    # Scrapes prices from an e-commerce page.\n    pass\n")
        with open(os.path.join(dir_b, "main.py"), "w", encoding="utf-8") as f:
            f.write("def train_image_classifier(dataset):\n    # Trains a CNN on labeled images.\n    pass\n")

        project_sources = [
            {"name": "price-scraper", "kind": "local", "location": dir_a},
            {"name": "image-classifier", "kind": "local", "location": dir_b},
        ]
        project_store, project_embedder, docs_store, docs_embedder = build_indices(
            TfidfEmbedder, project_sources=project_sources
        )

        sources = {c.source for c in project_store.chunks()}
        assert "project:price-scraper/main.py" in sources
        assert "project:image-classifier/main.py" in sources

        # This is the actual claim being tested: retrieval alone finds the
        # right PROJECT, not just the right file within one project.
        query_vector = project_embedder.embed(["I scraped e-commerce prices from web pages"])[0]
        assert project_store.search(query_vector, k=1)[0].chunk.source == "project:price-scraper/main.py"

        query_vector = project_embedder.embed(["I trained a CNN image classifier on a labeled dataset"])[0]
        assert project_store.search(query_vector, k=1)[0].chunk.source == "project:image-classifier/main.py"


def test_github_project_recovered_without_refetch_when_persisted():
    fake_chunks = [Chunk(text="def hello():\n    pass", source="someuser/some-repo/main.py")]
    project_sources = [{"name": "my-repo", "kind": "github", "location": "someuser/some-repo"}]

    with _TempPersistDir() as persist_dir:
        with patch("src.rag.index.fetch_github_repo_files", return_value=fake_chunks) as mock_fetch:
            build_indices(TfidfEmbedder, persist_path=persist_dir, project_sources=project_sources)
            assert mock_fetch.call_count == 1

            # Same persisted path, same project on a second call - should
            # recover from persistence, not fetch again.
            project_store, *_ = build_indices(TfidfEmbedder, persist_path=persist_dir, project_sources=project_sources)
            assert mock_fetch.call_count == 1

            sources = {c.source for c in project_store.chunks()}
            assert "project:my-repo:github:someuser/some-repo/main.py" in sources


def test_removed_project_is_dropped_from_persisted_store():
    fake_chunks = [Chunk(text="def hello():\n    pass", source="someuser/some-repo/main.py")]
    project_sources = [{"name": "my-repo", "kind": "github", "location": "someuser/some-repo"}]

    with _TempPersistDir() as persist_dir:
        with patch("src.rag.index.fetch_github_repo_files", return_value=fake_chunks):
            build_indices(TfidfEmbedder, persist_path=persist_dir, project_sources=project_sources)

        # Rebuild with an empty project list - the previously-fetched
        # content for "my-repo" should be dropped, not linger forever.
        project_store, *_ = build_indices(TfidfEmbedder, persist_path=persist_dir, project_sources=[])
        assert project_store.chunks() == []
