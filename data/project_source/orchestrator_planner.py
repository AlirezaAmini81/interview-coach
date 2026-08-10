"""
The planner turns (conversation history + new message + agent registry)
into an ordered plan: a list of {"agent": name, "instruction": text} steps.

A single-intent message produces one step. A compound request like "summarize
the RAG paper and then quiz me on it" produces two steps, executed in order -
this is what makes routing more than a single classification decision.
"""
from ..llm.provider import LLMProvider
from .registry import registry_as_prompt_text

SYSTEM_PROMPT = """You are the routing planner for a multi-agent research assistant.

Available agents:
{registry}

Given the conversation so far and the newest user message, decide which
agent(s) should handle the new message, in order. Most messages need exactly
one agent. Only use more than one step if the message genuinely asks for
multiple distinct actions (e.g. "summarize X and then quiz me on it").

Important: paper_qa is ONLY for questions about the actual content of the
loaded papers (orchestration, RAG, evaluation, tool use, prompt engineering).
Questions about the system itself - what it can do, how to use it, general
greetings or small talk - are general_chat, even if they mention words like
"system" or "papers" in passing. If you are not confident the message is
asking about specific paper content, choose general_chat.

Respond with ONLY a JSON object of this exact shape, nothing else:
{{"steps": [{{"agent": "<agent_name>", "instruction": "<what that agent should do>"}}]}}
"""


def create_plan(history: list[dict], user_message: str, llm: LLMProvider) -> list[dict]:
    system = SYSTEM_PROMPT.format(registry=registry_as_prompt_text())
    recent_history = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])
    prompt = f"Conversation so far:\n{recent_history}\n\nNewest user message: {user_message}"

    try:
        result = llm.complete_json([{"role": "user", "content": prompt}], system=system)
        steps = result.get("steps", [])
    except (ValueError, KeyError):
        # Small local models occasionally return malformed JSON. Don't crash
        # the app over a routing decision - fall back to plain chat.
        steps = []

    if not steps:
        # A confused planner should default to a plain chat response, not
        # force paper_qa to invent a "grounded" answer out of thin air.
        steps = [{"agent": "general_chat", "instruction": user_message}]

    return steps
