"""
CLI entry point. Run with `python -m src.main` from the project root.

Defaults to MockProvider so the whole system - mode routing, both RAG
indices, fact-checking, scoring, session tracking - runs and is inspectable
with no API key. Pass --provider ollama or --provider anthropic for real
routing decisions and real generated content.
"""
import argparse
import os

from dotenv import load_dotenv

from .agents.must_have_extractor_agent import extract_must_haves
from .llm.mock_provider import MockProvider
from .orchestrator.graph import build_graph
from .orchestrator.session_state import SessionState
from .rag.embeddings import TfidfEmbedder
from .rag.index import build_indices

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
CV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cv_placeholder.txt")
JOB_POSTING_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "job_posting.txt")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", ".chroma_data")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "anthropic", "ollama"], default="mock")
    parser.add_argument("--ollama-model", default="llama3.2:1b")
    parser.add_argument("--embedder", choices=["tfidf", "ollama"], default="tfidf")
    args = parser.parse_args()

    if args.provider == "anthropic":
        from .llm.langchain_provider import LangChainProvider

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set. See .env.example.")
        llm = LangChainProvider(model=DEFAULT_ANTHROPIC_MODEL, provider="anthropic", api_key=api_key)
        print("Using the Anthropic API via LangChain.\n")
    elif args.provider == "ollama":
        from .llm.langchain_provider import LangChainProvider

        llm = LangChainProvider(model=args.ollama_model, provider="ollama")
        print(f"Using local Ollama ({args.ollama_model}) via LangChain.\n")
    else:
        llm = MockProvider()
        print("Using MockProvider (no API key needed). Pass --provider ollama or "
              "--provider anthropic for real routing and answers.\n")

    embedder_factory = TfidfEmbedder
    if args.embedder == "ollama":
        from .rag.embeddings import OllamaEmbedder
        embedder_factory = OllamaEmbedder

    project_store, project_embedder, docs_store, docs_embedder = build_indices(embedder_factory, persist_path=CHROMA_PATH)
    graph = build_graph(llm, project_store, project_embedder, docs_store, docs_embedder)

    with open(CV_PATH, encoding="utf-8") as f:
        cv_text = f.read()
    if "[Replace this entire section" in cv_text:
        print("NOTE: data/cv_placeholder.txt is still the placeholder - questions "
              "and scoring won't be calibrated to your real background until you fill it in.\n")

    job_posting_text = ""
    if os.path.exists(JOB_POSTING_PATH):
        with open(JOB_POSTING_PATH, encoding="utf-8") as f:
            job_posting_text = f.read()
    must_haves = extract_must_haves(job_posting_text, llm)

    state = SessionState(must_haves=must_haves)
    print("Interview coach ready. Answer questions as they come, or ask to have "
          "something explained (e.g. 'explain LangGraph conditional edges'). "
          "Type 'quit' to exit.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue

        result = graph.invoke({
            "user_message": user_message,
            "cv_text": cv_text,
            "session_state": state,
            "response": "",
            "session_complete": False,
            "_mode": "",
        })

        print(f"\nCoach: {result['response']}\n")

        state.record_turn("user", user_message)
        state.record_turn("assistant", result["response"])

        if result.get("session_complete"):
            break


if __name__ == "__main__":
    main()
