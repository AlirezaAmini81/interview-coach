"""
Session state - the accumulated memory across the whole session. This is
what makes routing decisions adaptive turn to turn: which must-haves have
been covered, how you've scored so far, and the full Q&A history.

Kept as a plain dataclass with plain methods rather than anything fancier -
the whole point is that this stays easy to inspect and reason about.
"""
from dataclasses import dataclass, field

# The substantive, answerable-in-conversation must-haves from the posting.
# Eligibility/logistics items (enrollment, campus, language) are excluded -
# they're not something a mock interview can meaningfully probe.
MUST_HAVES = [
    "conversational agents knowledge",
    "LLM knowledge",
    "multi-agent systems knowledge",
    "Python programming ability",
    "data analysis / experiments / user studies experience",
    "scientific work experience",
]


@dataclass
class ScoredAnswer:
    topic: str
    question: str
    answer: str
    score: int
    justification: str
    kind: str = "behavioral"  # "behavioral" or "quiz"
    fact_check_verdict: str | None = None


@dataclass
class SessionState:
    history: list[dict] = field(default_factory=list)
    scored_answers: list[ScoredAnswer] = field(default_factory=list)
    covered_topics: set[str] = field(default_factory=set)
    # Defaults to the hardcoded MUST_HAVES above; pass an explicit list (see
    # agents/must_have_extractor_agent.py) to test against a real,
    # dynamically-parsed job posting instead.
    must_haves: list[str] = field(default_factory=lambda: list(MUST_HAVES))
    turn_count: int = 0
    pending_question: str | None = None
    pending_topic: str | None = None
    pending_kind: str | None = None  # "behavioral" or "quiz"
    pending_quiz_excerpts: str | None = None
    # Reference material a pending BEHAVIORAL question was grounded in, if
    # any was found (see graph.py's _advance_or_wrap_up) - lets scoring
    # check topic correctness against the same material, not just the
    # candidate's own project. None when nothing relevant was found for
    # the topic - a normal, expected case, not an error.
    pending_topic_excerpts: str | None = None
    probe_counts: dict = field(default_factory=dict)  # topic -> number of times probed

    def record_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def record_score(self, scored: ScoredAnswer) -> None:
        self.scored_answers.append(scored)
        self.turn_count += 1
        # Deliberately does NOT touch covered_topics - whether a topic counts
        # as "covered" is now a judgment call the graph makes (good score, or
        # probed enough times), not an automatic side effect of scoring.

    def uncovered_topics(self) -> list[str]:
        return [t for t in self.must_haves if t not in self.covered_topics]

    def coverage_fraction(self) -> float:
        return len(self.covered_topics) / len(self.must_haves) if self.must_haves else 1.0

    def average_score(self) -> float:
        if not self.scored_answers:
            return 0.0
        return sum(s.score for s in self.scored_answers) / len(self.scored_answers)
