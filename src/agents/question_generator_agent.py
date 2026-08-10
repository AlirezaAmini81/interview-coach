"""
Picks the next interview question, targeting an uncovered must-have from
the real job posting, informed by the user's real background so questions
aren't generic.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider

SYSTEM_PROMPT = (
    "You are a research-position interviewer. Ask ONE focused interview "
    "question that probes the given topic. Ground it in the candidate's "
    "actual background if relevant context is provided - ask about their "
    "real project, not a generic version of the topic. Ask only the "
    "question, no preamble."
)


@traceable(name="question_generator", run_type="chain")
def generate_question(topic: str, cv_text: str, history: list[dict], llm: LLMProvider) -> str:
    recent_history = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
    user_message = (
        f"Topic to probe: {topic}\n\n"
        f"Candidate's background:\n{cv_text}\n\n"
        f"Conversation so far:\n{recent_history}\n\n"
        f"Ask the next question."
    )
    return llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
