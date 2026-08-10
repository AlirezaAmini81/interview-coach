"""
Builds a REAL LangSmith dataset from the same test cases as
consistency_eval.py, then runs a REAL LangSmith experiment against it using
evaluate(..., num_repetitions=N) - so each repeated scoring run is a real,
inspectable run in your LangSmith dashboard, not just a local print.

Requires a real LangSmith account and API key (.env.example has the
variables). This will NOT work with MockProvider or without a LangSmith
API key - there's no local fallback, since the entire point is exercising
the real platform, not simulating it.

Usage:
    python -m eval.langsmith_consistency_eval --provider ollama --repetitions 5
    python -m eval.langsmith_consistency_eval --provider anthropic --repetitions 5

Honest note on the last step: `evaluate()` and `Client.create_dataset` /
`create_examples` are used exactly per the installed langsmith SDK's real
signatures (verified against the local package during development). The
exact column layout of `results.to_pandas()` was NOT verified against a
live LangSmith account (no key available in the build environment) - this
script prints it out on first run specifically so you can see the real
shape and extend the aggregation past what's shown here.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from src.agents import evaluator_agent

DATASET_NAME = "interview-coach-evaluator-consistency"

TEST_CASES = [
    {
        "topic": "multi-agent systems knowledge",
        "question": "How does your orchestrator decide which agent handles a request?",
        "answer": (
            "A planner LLM call reads a registry of agent descriptions and the "
            "conversation history, and returns which agent should handle the "
            "message as structured output. A separate dispatcher, which is "
            "plain code with no AI involved, matches that name to the right "
            "function."
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


def ensure_dataset(client: Client) -> str:
    """Create the dataset if it doesn't exist yet; reuse it if it does."""
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        return existing[0].id

    dataset = client.create_dataset(
        DATASET_NAME,
        description="Fixed (topic, question, answer) triples for measuring "
                     "whether the Evaluator agent scores the same answer "
                     "consistently across repeated runs.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"topic": c["topic"], "question": c["question"], "answer": c["answer"]}, "outputs": {}}
            for c in TEST_CASES
        ],
    )
    return dataset.id


def make_target(llm):
    def target(inputs: dict) -> dict:
        result = evaluator_agent.score_answer(
            inputs["topic"], inputs["question"], inputs["answer"], None, llm
        )
        return {"score": result.get("score"), "justification": result.get("justification")}
    return target


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "ollama"], required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    if not os.environ.get("LANGCHAIN_API_KEY"):
        raise SystemExit("LANGCHAIN_API_KEY is not set - this script requires a real LangSmith account.")

    if args.provider == "anthropic":
        from src.llm.langchain_provider import LangChainProvider
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set.")
        llm = LangChainProvider(model="claude-haiku-4-5-20251001", provider="anthropic", api_key=api_key)
    else:
        from src.llm.langchain_provider import LangChainProvider
        llm = LangChainProvider(model="llama3.2:1b", provider="ollama")

    client = Client()
    dataset_id = ensure_dataset(client)

    results = evaluate(
        make_target(llm),
        data=dataset_id,
        num_repetitions=args.repetitions,
        experiment_prefix="evaluator-consistency",
        client=client,
    )

    print(f"\nExperiment created: {results.experiment_name}")
    print(f"View it at: {results.url}\n")

    df = results.to_pandas()
    print("Raw results (first look at the real column layout - use this to")
    print("compute per-example score variance/agreement across repetitions):\n")
    print(df.head(20))


if __name__ == "__main__":
    main()
