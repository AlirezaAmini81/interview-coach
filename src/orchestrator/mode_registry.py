"""
Mode registry - what the top-level router chooses between. Same pattern as
the agent registry in the first project: structured descriptions the router
reads, rather than hardcoded logic.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    name: str
    description: str
    example_requests: list[str]


MODE_REGISTRY: list[ModeSpec] = [
    ModeSpec(
        name="behavioral",
        description=(
            "The user is answering (or ready to answer) an interview "
            "question about their background, project, or experience. "
            "This is also the default when the message doesn't clearly "
            "ask to have something explained."
        ),
        example_requests=[
            "My orchestrator uses a planner that reads a registry of agent descriptions...",
            "I built a RAG pipeline using a hand-rolled vector store.",
            "ready",
            "next question",
        ],
    ),
    ModeSpec(
        name="explain",
        description=(
            "The user is asking to have a concept or tool explained to "
            "them, or asking for a comparison between two tools/approaches. "
            "Not an interview answer - a genuine question directed at the "
            "system."
        ),
        example_requests=[
            "explain LangGraph conditional edges to me",
            "what is groundedness in RAG?",
            "compare LangGraph and CrewAI",
            "what's the difference between LangChain and LangGraph?",
        ],
    ),
    ModeSpec(
        name="quiz",
        description=(
            "The user is explicitly asking to be tested/quizzed on a "
            "specific tool or concept - a NEW quiz request, not an answer "
            "to a question already asked."
        ),
        example_requests=[
            "quiz me on RAG",
            "test me on LangGraph",
            "ask me something about multi-agent orchestration",
        ],
    ),
]


def registry_as_prompt_text() -> str:
    lines = []
    for spec in MODE_REGISTRY:
        examples = "; ".join(spec.example_requests)
        lines.append(f"- {spec.name}: {spec.description} Examples: {examples}")
    return "\n".join(lines)
