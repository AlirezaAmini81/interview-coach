"""
Runs the SAME answer through the Evaluator agent multiple times
independently and measures agreement - the "does your own scoring system
agree with itself" metric promised for this project.

Most meaningful with a real provider - MockProvider's evaluator always
returns a fixed score, so agreement will trivially be 100%. This script
still exists for mock mode as a plumbing check, but run it with
--provider ollama or --provider anthropic for a metric that means anything.

Usage:
    python -m eval.consistency_eval --provider ollama --runs 5
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.agents import evaluator_agent
from src.llm.mock_provider import MockProvider

# A fixed (question, answer) pair to re-score repeatedly - real content,
# not a placeholder, so a real provider has something substantive to judge.
TEST_CASES = [
    {
        "topic": "multi-agent systems knowledge",
        "question": "How does your orchestrator decide which agent handles a request?",
        "answer": (
            "A planner LLM call reads a registry of agent descriptions and the "
            "conversation history, and returns which agent should handle the "
            "message as structured JSON. A separate dispatcher, which is plain "
            "code with no AI involved, matches that name to the right function."
        ),
    },
    {
        "topic": "LLM knowledge",
        "question": "What's the difference between sparse and dense retrieval?",
        "answer": (
            "Sparse retrieval like TF-IDF matches on shared vocabulary between "
            "the query and documents. Dense retrieval uses a neural embedding "
            "model to represent meaning, so it can match paraphrased queries "
            "that don't share exact words."
        ),
    },
]


def run(llm, num_runs: int) -> None:
    for case in TEST_CASES:
        scores = []
        for _ in range(num_runs):
            result = evaluator_agent.score_answer(case["topic"], case["question"], case["answer"], None, llm)
            scores.append(int(result.get("score", 0)))

        print(f"\nTopic: {case['topic']}")
        print(f"  Scores across {num_runs} independent runs: {scores}")
        print(f"  Mean: {statistics.mean(scores):.2f}, stdev: {statistics.pstdev(scores):.2f}")
        print(f"  Exact agreement rate: {scores.count(max(set(scores), key=scores.count)) / num_runs:.0%}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "anthropic", "ollama"], default="mock")
    parser.add_argument("--runs", type=int, default=5)
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

    run(llm, args.runs)


if __name__ == "__main__":
    main()
