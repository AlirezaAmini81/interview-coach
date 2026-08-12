"""
Extracts the substantive, interview-testable must-haves from a pasted job
posting, replacing what used to be a hardcoded list - MUST_HAVES in
session_state.py is kept only as the fallback default (empty posting, or a
parse failure).

Excludes eligibility/logistics items (enrollment, location, language) the
same way the original hardcoded list did - a mock interview can't
meaningfully probe those.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..llm.schemas import MustHaveList
from ..orchestrator.session_state import MUST_HAVES

SYSTEM_PROMPT = (
    "You read a job posting and extract only the substantive, "
    "interview-testable requirements - things a mock interview could "
    "actually ask about, like domain knowledge, technical skills, or prior "
    "experience. Exclude eligibility/logistics items such as enrollment "
    "status, work location, visa/language requirements, or hours per week. "
    "Return each as a short topic phrase (e.g. 'multi-agent systems "
    "knowledge'), not a full sentence."
)


@traceable(name="must_have_extractor", run_type="chain")
def extract_must_haves(job_posting_text: str, llm: LLMProvider) -> list[str]:
    if not job_posting_text.strip():
        return list(MUST_HAVES)

    try:
        result = llm.complete_json(
            [{"role": "user", "content": job_posting_text}], system=SYSTEM_PROMPT, schema=MustHaveList
        )
        must_haves = result.get("must_haves")
        return must_haves if must_haves else list(MUST_HAVES)
    except (ValueError, KeyError):
        return list(MUST_HAVES)
