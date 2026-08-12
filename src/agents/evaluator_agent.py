"""
Scores an answer 1-5. Two functions sharing the same real ScoreResult
schema: score_answer (behavioral, takes the fact-checker's verdict into
account, and optionally real reference material about the topic itself)
and score_quiz_answer (grounded directly in the excerpts the quiz question
came from).
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..llm.schemas import ScoreResult

SYSTEM_PROMPT = (
    "You score an interview answer on a 1-5 scale for how well it "
    "demonstrates the given topic. If a fact-check verdict is provided and "
    "it is 'contradicted', the score must be 2 or lower regardless of how "
    "well-articulated the answer is - an inaccurate description of your "
    "own work is a real problem, not a style issue. If reference material "
    "about the topic itself is also provided, separately check whether the "
    "answer's claims about the topic are technically accurate against it - "
    "a confident but factually wrong description of the topic must also "
    "score 2 or lower, the same as a contradicted fact-check. These are two "
    "independent checks: one about the candidate's own project, one about "
    "the topic itself - either one failing caps the score."
)


@traceable(name="evaluator_behavioral", run_type="chain")
def score_answer(
    topic: str, question: str, answer: str, fact_check_verdict: str | None, llm: LLMProvider,
    topic_excerpts: str = "",
) -> dict:
    reference_block = f"\nReference material about the topic:\n{topic_excerpts}" if topic_excerpts else ""
    user_message = (
        f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}\n"
        f"Fact-check verdict: {fact_check_verdict or 'not applicable'}"
        f"{reference_block}"
    )
    try:
        result = llm.complete_json([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT, schema=ScoreResult)
        result.setdefault("score", 3)
        result.setdefault("justification", "")
        return result
    except (ValueError, KeyError):
        return {"score": 3, "justification": "evaluation call failed, default score"}


QUIZ_SYSTEM_PROMPT = (
    "You score a technical quiz answer on a 1-5 scale for correctness "
    "against the given reference excerpts - not general plausibility. An "
    "answer that sounds confident but contradicts the excerpts must score "
    "2 or lower."
)


@traceable(name="evaluator_quiz", run_type="chain")
def score_quiz_answer(question: str, answer: str, reference_excerpts: str, llm: LLMProvider) -> dict:
    user_message = f"Reference excerpts:\n{reference_excerpts}\n\nQuestion: {question}\nAnswer: {answer}"
    try:
        result = llm.complete_json([{"role": "user", "content": user_message}], system=QUIZ_SYSTEM_PROMPT, schema=ScoreResult)
        result.setdefault("score", 3)
        result.setdefault("justification", "")
        return result
    except (ValueError, KeyError):
        return {"score": 3, "justification": "evaluation call failed, default score"}
