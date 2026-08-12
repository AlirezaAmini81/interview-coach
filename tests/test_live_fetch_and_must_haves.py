"""
extract_topics is pure regex logic (no I/O) - tested directly. The
explainer's fallback path is tested with the live-fetch network call
monkeypatched to fail, so it stays offline and deterministic like the rest
of the suite - same reasoning as test_followup_and_quiz.py's ScriptedProvider.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import evaluator_agent, explainer_agent, fact_checker_agent
from src.agents.must_have_extractor_agent import extract_must_haves
from src.llm.mock_provider import MockProvider
from src.orchestrator.session_state import MUST_HAVES
from src.rag.embeddings import TfidfEmbedder
from src.rag.index import build_indices
from src.rag.live_fetch import extract_topics


def test_extract_topics_simple_question():
    assert extract_topics("explain LangGraph conditional edges") == ["LangGraph conditional edges"]


def test_extract_topics_vs_comparison():
    assert extract_topics("CrewAI vs AutoGen") == ["CrewAI", "AutoGen"]


def test_extract_topics_and_comparison():
    assert extract_topics("compare LangGraph and CrewAI") == ["LangGraph", "CrewAI"]


def test_extract_topics_strips_trailing_filler():
    # Regression: 'to me' left dangling on the end used to become part of
    # the search query ('docker to me'), which matched an unrelated
    # mega-popular repo (oh-my-zsh, which mentions docker as one of many
    # plugins) instead of an actual Docker repo.
    assert extract_topics("explain docker to me") == ["docker"]
    assert extract_topics("compare LangGraph and CrewAI for me") == ["LangGraph", "CrewAI"]


def test_must_have_extractor_falls_back_on_empty_posting():
    llm = MockProvider()
    assert extract_must_haves("", llm) == list(MUST_HAVES)


def test_must_have_extractor_returns_nonempty_list():
    llm = MockProvider()
    result = extract_must_haves("Some job posting about Python and multi-agent systems.", llm)
    assert result


def test_explainer_falls_back_honestly_when_fetch_unavailable():
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)
    llm = MockProvider()

    with patch("src.rag.live_fetch.fetch_github_readme", return_value=None):
        # Gibberish sharing zero vocabulary with the local corpus - scores
        # 0.0 locally, so this exercises the "fetch attempted, still nothing"
        # path rather than a real vocabulary-overlap coincidence.
        answer = explainer_agent.explain(
            "explain zzqxvbflibbergibbet12345", docs_store, docs_embedder, llm
        )

    assert "No relevant material found" in answer


def test_fact_checker_returns_not_applicable_for_irrelevant_answer():
    # "Docker lets you package an app with its dependencies" is a real,
    # true statement, but has nothing to do with this project's actual
    # code (a LangGraph-based interview coach) - the fact-checker should
    # say so explicitly via the relevance gate, not hand weak excerpts to
    # the LLM and hope it notices.
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)
    llm = MockProvider()

    result = fact_checker_agent.check_answer(
        "Docker lets you package an app with its dependencies so it runs the same everywhere.",
        project_store, project_embedder, llm,
    )
    assert result["verdict"] == "not_applicable"


def test_score_answer_accepts_topic_excerpts():
    llm = MockProvider()
    result = evaluator_agent.score_answer(
        "LangGraph", "Tell me about your experience with LangGraph.",
        "I used it to build a routing graph.", None, llm,
        topic_excerpts="[langgraph.txt] LangGraph represents an agentic workflow as a graph.",
    )
    assert "score" in result and "justification" in result
