"""
Mode router - decides which of three flows handles the incoming message.

Takes an explicit hint about whether something is currently pending, since
that materially changes how ambiguous messages should be read: a technical-
sounding message is more likely an ANSWER to a pending quiz question than a
brand new quiz request, and the router can't know that from the message
text alone.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..llm.schemas import ModeDecision
from .mode_registry import registry_as_prompt_text

SYSTEM_PROMPT = """You are the mode router for an interview-prep coaching system.

Available modes:
{registry}

{pending_hint}

Given the newest user message, decide the mode."""

PENDING_HINT = (
    "IMPORTANT: a question is currently pending an answer. If this message "
    "reads like an attempt to answer it - even a technical-sounding one - "
    "classify it as 'behavioral', not 'explain' or 'quiz'. Only use "
    "'explain' or 'quiz' if the message is clearly a new, separate request."
)


@traceable(name="mode_router", run_type="chain")
def decide_mode(user_message: str, llm: LLMProvider, has_pending: bool = False) -> str:
    system = SYSTEM_PROMPT.format(
        registry=registry_as_prompt_text(),
        pending_hint=PENDING_HINT if has_pending else "",
    )
    try:
        result = llm.complete_json(
            [{"role": "user", "content": user_message}], system=system, schema=ModeDecision
        )
        mode = result.get("mode")
    except (ValueError, KeyError):
        mode = None

    return mode if mode in ("behavioral", "explain", "quiz") else "behavioral"
