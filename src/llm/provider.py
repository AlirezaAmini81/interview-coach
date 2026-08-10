"""
Abstract interface every LLM provider implements.

Two methods only:
- complete: free-text generation (used by the agents to answer, summarize, quiz)
- complete_json: structured generation (used by the planner to decide routing)

Keeping these separate makes the planner's output type explicit instead of
hoping the model's free text happens to parse as JSON.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        """Return the model's free-text reply to a list of {"role", "content"} messages."""
        raise NotImplementedError

    @abstractmethod
    def complete_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Return the model's reply as a dict matching `schema` if provided.
        Callers should pass a real Pydantic schema wherever they know the
        expected shape - providers that support real structured output
        (see langchain_provider.py) use it directly instead of parsing text."""
        raise NotImplementedError
