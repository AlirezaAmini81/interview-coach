"""
Picks the next interview question, targeting an uncovered must-have from
the real job posting, informed by the user's real background so questions
aren't generic.

If real reference material about the topic itself was found (see
graph.py's _advance_or_wrap_up, which tries live_fetch.retrieve_with_fetch
before calling this), it gets passed in as topic_excerpts so the question
can be more specific and technically accurate - without turning into a
knowledge quiz, which is a separate mode with its own agent.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider

SYSTEM_PROMPT = (
    "You are a research-position interviewer. Ask ONE focused interview "
    "question that probes the given topic. Ground it in the candidate's "
    "actual background if relevant context is provided - ask about their "
    "real project, not a generic version of the topic. If reference "
    "material about the topic itself is also provided, use it to make the "
    "question more specific and technically accurate - but keep it framed "
    "as asking about the candidate's own experience, not a knowledge quiz. "
    "Ask only the question, no preamble."
)


@traceable(name="question_generator", run_type="chain")
def generate_question(
    topic: str, cv_text: str, history: list[dict], llm: LLMProvider, topic_excerpts: str = ""
) -> str:
    recent_history = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
    reference_block = f"\n\nReference material about the topic:\n{topic_excerpts}" if topic_excerpts else ""
    user_message = (
        f"Topic to probe: {topic}\n\n"
        f"Candidate's background:\n{cv_text}\n\n"
        f"Conversation so far:\n{recent_history}"
        f"{reference_block}\n\n"
        f"Ask the next question."
    )
    return llm.complete([{"role": "user", "content": user_message}], system=SYSTEM_PROMPT)
