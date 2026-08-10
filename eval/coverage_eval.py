"""
Runs a full scripted session and checks: does the system actually cover
every must-have from the real job posting within the turn budget, or does
it stall? This is the coverage metric promised for this project.

Usage:
    python -m eval.coverage_eval
    python -m eval.coverage_eval --provider ollama
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.llm.mock_provider import MockProvider
from src.orchestrator.graph import build_graph
from src.orchestrator.session_state import MUST_HAVES, SessionState
from src.rag.embeddings import TfidfEmbedder
from src.rag.index import build_indices

# Generic but non-empty scripted answers - enough for the graph to score
# something real without needing an interactive human for this check.
SCRIPTED_ANSWERS = [
    "ready",
    "I built a chatbot that routes questions to different specialized agents.",
    "I've used Claude and GPT-4 APIs directly, and read about how they handle context.",
    "My orchestrator has a planner, a dispatcher, and four specialist agents.",
    "I'm comfortable with Python, including type hints and dataclasses.",
    "I ran an evaluation script comparing routing accuracy before and after a prompt fix.",
    "I documented a bug I found, why it happened, and how I fixed it, with before/after numbers.",
]


def run(llm) -> None:
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)
    graph = build_graph(llm, project_store, project_embedder, docs_store, docs_embedder)

    state = SessionState()
    cv_text = "Placeholder CV for automated coverage testing."
    result = {}

    for message in SCRIPTED_ANSWERS:
        result = graph.invoke({
            "user_message": message,
            "cv_text": cv_text,
            "session_state": state,
            "response": "",
            "session_complete": False,
            "_mode": "",
        })
        state.record_turn("user", message)
        state.record_turn("assistant", result["response"])
        if result.get("session_complete"):
            break

    covered = len(state.covered_topics)
    total = len(MUST_HAVES)
    print(f"Coverage: {covered}/{total} must-haves")
    for topic in MUST_HAVES:
        status = "COVERED" if topic in state.covered_topics else "MISSED"
        print(f"  [{status}] {topic}")
    print(f"Turns used: {state.turn_count} (budget allows early completion)")
    print(f"Session completed naturally: {result.get('session_complete', False)}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "anthropic", "ollama"], default="mock")
    args = parser.parse_args()

    if args.provider == "anthropic":
        from src.llm.langchain_provider import LangChainProvider
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set.")
        llm = LangChainProvider(model="claude-haiku-4-5-20251001", provider="anthropic", api_key=api_key)
    elif args.provider == "ollama":
        from src.llm.langchain_provider import LangChainProvider
        llm = LangChainProvider(model="llama3.2:1b", provider="ollama")
    else:
        llm = MockProvider()

    run(llm)


if __name__ == "__main__":
    main()
