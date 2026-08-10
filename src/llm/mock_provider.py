"""
Deterministic fake provider - no API key, no network call.

This project has several DIFFERENT structured-JSON call sites (mode router,
evaluator, fact-checker) unlike the first project's single planner. Since a
mock has no real understanding, it sniffs the system prompt text to guess
which kind of call is being made, then returns a plausible-shaped fake
answer for that specific call. This is naive by design - swap in
LangChainProvider (see langchain_provider.py) for real decisions via real
structured output.

Accepts a `schema` argument for interface compatibility with real providers
but ignores it - mock mode doesn't do real structured output, just returns
a plain dict shaped correctly by the keyword heuristics below.
"""
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from .provider import LLMProvider


class MockProvider(LLMProvider):
    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        last_user = messages[-1]["content"] if messages else ""
        return f"[mock completion - would answer]: {last_user[:200]}"

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        system_lower = (system or "").lower()
        last_user = (messages[-1]["content"] if messages else "").lower()

        if "which mode" in system_lower or "mode router" in system_lower:
            # naive: looks like a question about a tool/concept -> explain mode
            quiz_signals = ["quiz me", "test me", "ask me something"]
            explain_signals = ["explain", "what is", "how does", "compare", " vs ", "difference between"]
            if any(s in last_user for s in quiz_signals):
                mode = "quiz"
            elif any(s in last_user for s in explain_signals):
                mode = "explain"
            else:
                mode = "behavioral"
            return {"mode": mode, "reasoning": "mock: keyword heuristic"}

        if "not_applicable" in system_lower:
            return {"verdict": "not_applicable", "reasoning": "mock: no implementation claim detected"}

        if "score" in system_lower and "1-5" in system_lower:
            return {"score": 3, "justification": "mock: placeholder mid-range score"}

        # Unknown structured call - fail safe rather than guess wrong
        return {}
