"""
Aggregates the session into a final report: mostly plain code (coverage,
average score, weak topics are all just arithmetic over SessionState), with
one LLM call for a short qualitative summary a spreadsheet can't produce.

Behavioral and quiz scores are reported separately - they measure different
things (interview-answer quality vs technical-knowledge correctness) and
shouldn't be averaged together.
"""
from langsmith import traceable

from ..llm.provider import LLMProvider
from ..orchestrator.session_state import SessionState

SYSTEM_PROMPT = (
    "You write a brief, honest, encouraging-but-direct closing summary of "
    "an interview practice session, given the scored answers. Point out "
    "the single weakest area and one concrete thing to review before the "
    "real interview. 3-4 sentences."
)


@traceable(name="report_synthesizer", run_type="chain")
def build_report(state: SessionState, llm: LLMProvider) -> str:
    behavioral = [s for s in state.scored_answers if s.kind == "behavioral"]
    quiz = [s for s in state.scored_answers if s.kind == "quiz"]

    lines = [
        f"Topics covered: {len(state.covered_topics)}/{len(state.covered_topics) + len(state.uncovered_topics())}",
    ]
    if behavioral:
        avg = sum(s.score for s in behavioral) / len(behavioral)
        lines.append(f"Behavioral average: {avg:.1f}/5 ({len(behavioral)} answers)")
    if quiz:
        avg = sum(s.score for s in quiz) / len(quiz)
        lines.append(f"Quiz average: {avg:.1f}/5 ({len(quiz)} questions)")
    if state.uncovered_topics():
        lines.append(f"Not yet covered: {', '.join(state.uncovered_topics())}")

    for s in behavioral:
        probes = state.probe_counts.get(s.topic, 1)
        probe_note = f" (took {probes} attempts)" if probes > 1 else ""
        lines.append(f"- [behavioral: {s.topic}] {s.score}/5{probe_note} - {s.justification}")
    for s in quiz:
        lines.append(f"- [quiz: {s.topic}] {s.score}/5 - {s.justification}")

    stats_block = "\n".join(lines)
    summary = llm.complete(
        [{"role": "user", "content": f"Session results:\n{stats_block}"}], system=SYSTEM_PROMPT
    )

    return f"{stats_block}\n\nSummary:\n{summary}"
