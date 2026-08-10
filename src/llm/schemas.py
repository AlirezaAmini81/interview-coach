"""
Real schemas for LangChain's .with_structured_output() - the model is
constrained to return exactly this shape (via tool-calling or Claude's
native structured-output feature under the hood), rather than being asked
nicely to produce JSON and having us parse a string and hope.
"""
from pydantic import BaseModel, Field


class ModeDecision(BaseModel):
    """The router's decision about which mode should handle the message."""
    mode: str = Field(description="Exactly one of: behavioral, explain, quiz")
    reasoning: str = Field(description="One short sentence explaining the choice")


class FactCheckVerdict(BaseModel):
    """Whether a self-description claim is supported by real source excerpts."""
    verdict: str = Field(description="Exactly one of: supported, contradicted, not_applicable")
    reasoning: str = Field(description="One or two sentences")


class ScoreResult(BaseModel):
    """A 1-5 score with justification - used for both behavioral and quiz answers."""
    score: int = Field(description="Integer score from 1 to 5", ge=1, le=5)
    justification: str = Field(description="One sentence")
