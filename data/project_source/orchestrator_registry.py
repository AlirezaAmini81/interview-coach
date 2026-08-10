"""
The agent registry: a structured description of what each agent can do.

This is what makes the orchestrator extensible. The planner never has
if/else logic like `if "quiz" in message: call quiz_agent()` - instead it is
handed this list of specs and an LLM decides, based on the descriptions,
which agent(s) fit the request. Adding a new agent means adding one AgentSpec
here and one function in agents/ - the planner's prompt updates automatically
because it always reads from this registry.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    example_requests: list[str]


AGENT_REGISTRY: list[AgentSpec] = [
    AgentSpec(
        name="general_chat",
        description=(
            "Handles greetings, small talk, meta-questions about the system "
            "itself (what it can do, how to use it), and anything NOT asking "
            "about specific content from the loaded papers. This is the "
            "default when a message isn't clearly a paper question - never "
            "force paper_qa onto a greeting or off-topic message."
        ),
        example_requests=[
            "hi",
            "what can you help me with?",
            "what can I do with this system?",
            "thanks!",
        ],
    ),
    AgentSpec(
        name="paper_qa",
        description=(
            "Answers factual questions about the actual CONTENT of the "
            "loaded research papers, grounded in retrieved excerpts. Use "
            "only when the user is asking what a specific paper says, "
            "explains, or claims about orchestration, RAG, evaluation, tool "
            "use, or prompt engineering - not questions about the system."
        ),
        example_requests=[
            "What is retrieval-augmented generation?",
            "How does the orchestration paper describe rule-based routing?",
            "What does the evaluation paper say about multi-turn coherence?",
        ],
    ),
    AgentSpec(
        name="summarizer",
        description=(
            "Produces a concise summary of a paper, or of what has been "
            "discussed so far in the conversation."
        ),
        example_requests=[
            "Summarize the RAG paper.",
            "Can you sum up what we've covered so far?",
        ],
    ),
    AgentSpec(
        name="quiz",
        description=(
            "Generates comprehension questions to test understanding of "
            "material already discussed in this conversation."
        ),
        example_requests=[
            "Quiz me on that.",
            "Test my understanding of orchestration.",
        ],
    ),
]


def registry_as_prompt_text() -> str:
    """Render the registry as text the planner LLM reads to decide routing."""
    lines = []
    for spec in AGENT_REGISTRY:
        examples = "; ".join(spec.example_requests)
        lines.append(f"- {spec.name}: {spec.description} Examples: {examples}")
    return "\n".join(lines)
