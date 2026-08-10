"""
The default MockProvider always scores 3/5, which never crosses the <3
follow-up threshold - so testing the follow-up logic needs a provider that
can be made to return a low score on purpose.
"""
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.provider import LLMProvider
from src.orchestrator.graph import build_graph
from src.orchestrator.session_state import SessionState
from src.rag.embeddings import TfidfEmbedder
from src.rag.index import build_indices


class ScriptedProvider(LLMProvider):
    """A mock whose evaluator score is controllable per-call, for testing
    branch logic the default MockProvider can't reach."""

    def __init__(self, scores: list[int]):
        self.scores = scores  # popped in order, one per evaluator call
        self.call_count = 0

    def complete(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        return f"[scripted]: {messages[-1]['content'][:50]}"

    def complete_json(self, messages: List[Dict[str, str]], system: Optional[str] = None, schema=None) -> Dict[str, Any]:
        system_lower = (system or "").lower()
        last_user = (messages[-1]["content"] if messages else "").lower()

        if "which mode" in system_lower or "mode router" in system_lower:
            if "quiz me" in last_user:
                return {"mode": "quiz"}
            return {"mode": "behavioral"}

        if "not_applicable" in system_lower:
            return {"verdict": "not_applicable", "reasoning": "scripted: n/a"}

        if "score" in system_lower and "1-5" in system_lower:
            score = self.scores[self.call_count] if self.call_count < len(self.scores) else 3
            self.call_count += 1
            return {"score": score, "justification": "scripted low score"}

        return {}


def _fresh_graph(llm):
    project_store, project_embedder, docs_store, docs_embedder = build_indices(TfidfEmbedder)
    return build_graph(llm, project_store, project_embedder, docs_store, docs_embedder)


def _invoke(graph, state, message):
    result = graph.invoke({
        "user_message": message, "cv_text": "test cv", "session_state": state,
        "response": "", "session_complete": False, "_mode": "",
    })
    state.record_turn("user", message)
    state.record_turn("assistant", result["response"])
    return result


def test_low_score_triggers_followup_not_next_topic():
    llm = ScriptedProvider(scores=[2])  # one weak answer
    graph = _fresh_graph(llm)
    state = SessionState()

    _invoke(graph, state, "ready")  # gets first question
    first_topic = state.pending_topic
    result = _invoke(graph, state, "a weak answer")

    # Should still be pending on the SAME topic (follow-up), not covered yet.
    assert state.pending_kind == "behavioral"
    assert state.pending_topic == first_topic
    assert first_topic not in state.covered_topics
    assert state.probe_counts[first_topic] == 1


def test_two_weak_scores_moves_on_anyway():
    llm = ScriptedProvider(scores=[2, 2])  # weak, then weak again
    graph = _fresh_graph(llm)
    state = SessionState()

    _invoke(graph, state, "ready")
    first_topic = state.pending_topic
    _invoke(graph, state, "weak answer one")  # triggers follow-up
    _invoke(graph, state, "weak answer two")  # second weak answer on same topic

    # After MAX_PROBES_PER_TOPIC (2), must move on regardless of score.
    assert first_topic in state.covered_topics
    assert state.pending_topic != first_topic


def test_quiz_mode_start_and_answer():
    llm = ScriptedProvider(scores=[4])
    graph = _fresh_graph(llm)
    state = SessionState()

    result = _invoke(graph, state, "quiz me on RAG")
    assert state.pending_kind == "quiz"
    assert state.pending_quiz_excerpts  # real excerpts were retrieved and stored

    _invoke(graph, state, "my answer to the quiz question")
    assert state.pending_kind is None  # cleared after scoring
    quiz_scores = [s for s in state.scored_answers if s.kind == "quiz"]
    assert len(quiz_scores) == 1
    assert quiz_scores[0].score == 4
