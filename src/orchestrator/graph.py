"""
The orchestrator graph. Real conditional branch points:

  mode_router --[behavioral|explain|quiz]--> behavioral / explain / quiz
  behavioral  --[complete|continue]--> report / END

"behavioral" is really "the answer/default handler": it deals with whatever
is currently pending (a behavioral question OR a quiz question - checked via
pending_kind), or asks a fresh behavioral question if nothing is pending.
Starting a NEW quiz mid-interview overwrites any pending behavioral question
(a known, documented scope limitation, not an oversight).

Each call to graph.invoke() handles exactly one incoming message - the
multi-turn session loop happens across repeated invoke() calls driven by
main.py, with SessionState carried forward each time.
"""
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import (
    evaluator_agent,
    explainer_agent,
    fact_checker_agent,
    followup_prober_agent,
    question_generator_agent,
    report_synthesizer,
    tech_quizzer_agent,
)
from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore
from .mode_router import decide_mode
from .session_state import ScoredAnswer, SessionState

MAX_TURNS = 10
MAX_PROBES_PER_TOPIC = 2
LOW_SCORE_THRESHOLD = 3

# Deliberately stricter than live_fetch.RELEVANCE_THRESHOLD (0.2), and
# deliberately LOCAL-ONLY (no live-fetch attempt) - unlike explain/quiz,
# where a bad match just means "nothing found, say so" (safe), a bad match
# here means WRONG material gets silently used to write/score a question,
# which is worse than no grounding at all. Verified empirically: abstract
# must-have phrases like "scientific work experience" scored 0.253 against
# totally unrelated LangChain content - the shared word was generic filler
# ("work"), not a real signal - while genuinely relevant matches like
# "multi-agent systems knowledge" (0.555) and "conversational agents
# knowledge" (0.364, correctly matching on "agents") sit comfortably above
# 0.3. No live-fetch here also matters beyond cost: docs_store is the same
# persistent store explain/quiz use (see rag/live_fetch.py's Chroma
# persistence) - a spurious fetch for a non-tool phrase would pollute it
# permanently, not just for this one question.
BEHAVIORAL_GROUNDING_THRESHOLD = 0.3


class CoachState(TypedDict):
    user_message: str
    cv_text: str
    session_state: SessionState
    response: str
    session_complete: bool
    _mode: str


def build_graph(
    llm: LLMProvider,
    project_store: VectorStore,
    project_embedder: Embedder,
    docs_store: VectorStore,
    docs_embedder: Embedder,
):
    def mode_router_node(state: CoachState) -> dict[str, Any]:
        has_pending = bool(state["session_state"].pending_question)
        mode = decide_mode(state["user_message"], llm, has_pending=has_pending)
        return {"_mode": mode}

    def route_after_mode_router(state: CoachState) -> str:
        return state.get("_mode", "behavioral")

    def _advance_or_wrap_up(s: SessionState, feedback: str, cv_text: str) -> dict[str, Any]:
        """Shared tail: after a topic is fully resolved, either ask the next
        question or decide the session is complete."""
        if not s.uncovered_topics() or s.turn_count >= MAX_TURNS:
            wrap_up = (feedback + "\n\n" if feedback else "") + "That covers what I can meaningfully ask - wrapping up."
            return {"response": wrap_up, "session_complete": True}

        next_topic = s.uncovered_topics()[0]

        # Ground the question in real LOCAL material if there's confidently
        # relevant coverage - local search only, no live-fetch (see
        # BEHAVIORAL_GROUNDING_THRESHOLD above for why). Degrades cleanly to
        # "" for topics with nothing confidently relevant - the common case
        # for the abstract default must-haves, not a bug.
        topic_vector = docs_embedder.embed([next_topic])[0]
        topic_results = docs_store.search(topic_vector, k=3)
        topic_excerpts = ""
        if topic_results and topic_results[0].score >= BEHAVIORAL_GROUNDING_THRESHOLD:
            topic_excerpts = "\n\n".join(f"[{r.chunk.source}] {r.chunk.text}" for r in topic_results)

        question = question_generator_agent.generate_question(
            next_topic, cv_text, s.history, llm, topic_excerpts=topic_excerpts
        )
        s.pending_question, s.pending_topic, s.pending_kind = question, next_topic, "behavioral"
        s.pending_topic_excerpts = topic_excerpts or None

        response = (feedback + "\n\n" if feedback else "") + question
        return {"response": response, "session_complete": False}

    def behavioral_node(state: CoachState) -> dict[str, Any]:
        s = state["session_state"]

        if s.pending_kind == "behavioral" and s.pending_question:
            topic = s.pending_topic
            fc = fact_checker_agent.check_answer(state["user_message"], project_store, project_embedder, llm)
            ev = evaluator_agent.score_answer(
                topic, s.pending_question, state["user_message"], fc.get("verdict"), llm,
                topic_excerpts=s.pending_topic_excerpts or "",
            )
            score = int(ev.get("score", 3))

            s.record_score(ScoredAnswer(
                topic=topic, question=s.pending_question, answer=state["user_message"],
                score=score, justification=ev.get("justification", ""), kind="behavioral",
                fact_check_verdict=fc.get("verdict"),
            ))
            s.probe_counts[topic] = s.probe_counts.get(topic, 0) + 1

            feedback = f"Scored {score}/5 - {ev.get('justification', '')}"
            if fc.get("verdict") == "contradicted":
                feedback += f" (fact-check: {fc.get('reasoning', '')})"

            if score < LOW_SCORE_THRESHOLD and s.probe_counts[topic] < MAX_PROBES_PER_TOPIC:
                # Weak answer, topic not yet probed twice - dig deeper instead of moving on.
                followup = followup_prober_agent.generate_followup(
                    topic, s.pending_question, state["user_message"], ev.get("justification", ""), llm
                )
                s.pending_question, s.pending_kind = followup, "behavioral"  # topic (pending_topic) unchanged
                return {"response": f"{feedback}\n\n{followup}", "session_complete": False}

            # Either scored well, or already probed twice - move on for real.
            s.covered_topics.add(topic)
            s.pending_question, s.pending_topic, s.pending_kind, s.pending_topic_excerpts = None, None, None, None
            return _advance_or_wrap_up(s, feedback, state["cv_text"])

        if s.pending_kind == "quiz" and s.pending_question:
            ev = evaluator_agent.score_quiz_answer(s.pending_question, state["user_message"], s.pending_quiz_excerpts, llm)
            score = int(ev.get("score", 3))
            s.record_score(ScoredAnswer(
                topic=s.pending_topic or "quiz", question=s.pending_question, answer=state["user_message"],
                score=score, justification=ev.get("justification", ""), kind="quiz",
            ))
            s.pending_question, s.pending_topic, s.pending_kind, s.pending_quiz_excerpts = None, None, None, None
            return {"response": f"Scored {score}/5 - {ev.get('justification', '')}", "session_complete": False}

        # Nothing pending at all - this is either session start or the user said "ready"/similar.
        return _advance_or_wrap_up(s, "", state["cv_text"])

    def explain_node(state: CoachState) -> dict[str, Any]:
        answer = explainer_agent.explain(state["user_message"], docs_store, docs_embedder, llm)
        s = state["session_state"]
        if s.pending_kind == "behavioral" and s.pending_question:
            # Explain mode never touches pending state (unlike quiz mode,
            # which overwrites it - see behavioral_node) - but the question
            # scrolled out of view, so remind the user it's still open.
            answer += f"\n\n(Still waiting on: {s.pending_question})"
        return {"response": answer, "session_complete": False}

    def quiz_node(state: CoachState) -> dict[str, Any]:
        s = state["session_state"]
        question, excerpts = tech_quizzer_agent.generate_quiz_question(state["user_message"], docs_store, docs_embedder, llm)
        s.pending_question, s.pending_kind, s.pending_quiz_excerpts = question, "quiz", excerpts
        s.pending_topic = state["user_message"][:40]
        return {"response": question, "session_complete": False}

    def report_node(state: CoachState) -> dict[str, Any]:
        report = report_synthesizer.build_report(state["session_state"], llm)
        return {"response": state["response"] + "\n\n" + report}

    def route_after_behavioral(state: CoachState) -> str:
        return "report" if state.get("session_complete") else "end"

    graph = StateGraph(CoachState)
    graph.add_node("mode_router", mode_router_node)
    graph.add_node("behavioral", behavioral_node)
    graph.add_node("explain", explain_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("mode_router")
    graph.add_conditional_edges(
        "mode_router", route_after_mode_router,
        {"behavioral": "behavioral", "explain": "explain", "quiz": "quiz"},
    )
    graph.add_conditional_edges("behavioral", route_after_behavioral, {"report": "report", "end": END})
    graph.add_edge("explain", END)
    graph.add_edge("quiz", END)
    graph.add_edge("report", END)

    return graph.compile()
