"""
Real LangChain-backed provider. Replaces what used to be two separate
hand-rolled provider classes (one per SDK) with ONE implementation, because
that's the actual point of init_chat_model: which provider you're talking
to becomes a string you pass in, not a different class to import.

Structured output uses .with_structured_output(schema) with real Pydantic
models (see schemas.py) - the model is constrained to the shape via
tool-calling or the provider's native structured-output feature, not asked
nicely to produce JSON that we then parse and hope is valid.
"""
import json
import re
from typing import Any, Dict, List, Optional, Type

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .provider import LLMProvider


class LangChainProvider(LLMProvider):
    def __init__(self, model: str, provider: str, **kwargs):
        """
        model: the model name, e.g. "claude-haiku-4-5-20251001" or "llama3.2:1b"
        provider: e.g. "anthropic" or "ollama" - passed explicitly as
            model_provider rather than a combined "provider:model" string,
            since Ollama model names already contain a colon (llama3.2:1b),
            which would collide with init_chat_model's own prefix separator.
        """
        self.chat = init_chat_model(model=model, model_provider=provider, **kwargs)

    def _to_lc_messages(self, messages: List[Dict[str, str]], system: Optional[str]):
        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        for m in messages:
            lc_messages.append(HumanMessage(content=m["content"]))
        return lc_messages

    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        response = self.chat.invoke(self._to_lc_messages(messages, system))
        return response.content

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        if schema is not None:
            # The real thing: the model is constrained to this shape by
            # LangChain, not just asked to produce JSON in its text reply.
            structured_chat = self.chat.with_structured_output(schema)
            result = structured_chat.invoke(self._to_lc_messages(messages, system))
            return result.model_dump()

        # Fallback for the rare caller with no schema in hand: prompt for
        # JSON and parse it - the old, weaker approach, kept only as a safety net.
        json_instruction = "Respond with ONLY a valid JSON object. No markdown fences, no prose."
        full_system = f"{system}\n\n{json_instruction}" if system else json_instruction
        response = self.chat.invoke(self._to_lc_messages(messages, full_system))
        cleaned = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {response.content!r}") from exc
