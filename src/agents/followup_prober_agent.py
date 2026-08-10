"""
Generates a sharper follow-up question on the SAME topic when an answer
scored low - this is the adaptive piece that was missing from the MVP.
Sees the actual weak answer and feedback, not just the topic name, so the
follow-up targets the specific gap rather than repeating the original
question.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider

SYSTEM_PROMPT = (
    "You are a research-position interviewer. The candidate's previous "
    "answer on this topic was weak. Ask ONE sharper, more specific "
    "follow-up question that targets the actual gap in their answer - "
    "don't just repeat the original question. Ask only the question, no "
    "preamble."
)


@traceable(name="followup_prober", run_type="chain")
def generate_followup(topic: str, previous_question: str, weak_answer: str, feedback: str, llm: LLMProvider) -> str:
    user_message = (
        f"Topic: {topic}\n"
        f"Original question: {previous_question}\n"
        f"Candidate's weak answer: {weak_answer}\n"
        f"Why it was weak: {feedback}\n\n"
        f"Ask a sharper follow-up."
    )
    return llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
