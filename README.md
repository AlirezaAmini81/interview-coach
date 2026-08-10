# Interview coach

A multi-agent system for practicing an interview: it asks real questions
targeting the actual must-haves from a real job posting, scores your
answers, checks self-description claims against your real project's source
code via RAG, and separately answers your own questions about tools/concepts
(with real comparisons) grounded in a second, independent RAG corpus.

## Setup

```bash
pip install -r requirements.txt
```

**Before running:** replace `data/cv_placeholder.txt` with your real
background - questions and scoring are generic until you do.

Three provider options, none requiring money:
- **mock** (default) - no setup, canned text, real routing/RAG/scoring logic
- **ollama** - free, local, real answers (`ollama pull llama3.2:1b`)
- **anthropic** - paid API, best answers (new accounts get free trial credit)

## Running it

```bash
python -m src.main                        # mock
python -m src.main --provider ollama       # free, local, real
python -m src.main --provider anthropic    # paid API
```

Try both interaction modes:
- Just answer as questions come - this is Behavioral mode.
- At any point, ask something like `explain LangGraph conditional edges` or
  `compare LangGraph and CrewAI` - this is Explain mode, and works even
  with a question still pending; it'll come back to that question after.

## Running the evals

```bash
python -m eval.coverage_eval                          # does it cover all must-haves?
python -m eval.consistency_eval --provider ollama --runs 5   # local, does the evaluator agree with itself?
python -m eval.langsmith_consistency_eval --provider anthropic --repetitions 5  # same question, as a real LangSmith experiment
```

Consistency is only meaningful with a real provider - mock's evaluator
always returns the same score, so it trivially "agrees" with itself.

## Running the tests

```bash
python -m pytest tests/
```

## Architecture

```
src/
  llm/            mock / ollama / anthropic providers (same pattern as before)
  rag/            embeddings + vector store, but TWO separate indices now
  agents/         question_generator, fact_checker, evaluator, explainer, report_synthesizer
  orchestrator/
    session_state.py    tracks scores, covered topics, pending question
    mode_registry.py     what the router chooses between
    mode_router.py         the router itself
    graph.py                 LangGraph wiring, two real conditional branch points
data/
  job_posting.txt        the real OrKA posting
  cv_placeholder.txt      replace with your real background
  project_source/          real files copied from research-assistant-orchestrator
  tool_docs/                 original writeups on LangGraph/LangChain/LangSmith/RAG/multi-agent
eval/               coverage_eval.py, consistency_eval.py
```

## Where each concept actually lives

- **Two genuinely separate RAG indices**: `project_source/` (real code,
  used by the Fact-Checker to verify self-description) and `tool_docs/`
  (concept explainers, used by the Explainer and Tech Quizzer).
  `tests/test_indices.py` proves they don't bleed into each other.
- **Real LangGraph branching**: `orchestrator/graph.py` has genuine
  conditional edges - mode routing (behavioral/explain/quiz) and completion
  routing (another question vs wrap up to report). The multi-question
  "loop" across a session happens across repeated `graph.invoke()` calls
  driven by `main.py`, with `SessionState` carried forward each time - same
  honest architecture as the first project, not a fully autonomous cycle.
- **Adaptive follow-up**: a score below 3 routes to `followup_prober_agent`
  instead of the next topic - up to 2 attempts per topic before moving on
  regardless. `tests/test_followup_and_quiz.py` proves both branches.
- **Bidirectional tool Q&A**: `explainer_agent` (you ask, it answers,
  grounded, with real comparisons) and `tech_quizzer_agent` (it asks, you
  answer, scored against the exact excerpts the question came from) -
  mirror images of each other, both grounded in the same real docs corpus.
- **Real LangChain use** (`llm/langchain_provider.py`): a single provider
  class backed by `init_chat_model`, so Anthropic vs Ollama is a config
  value (`provider="anthropic"` vs `provider="ollama"`), not two different
  classes to maintain. Every structured-output call site (mode router,
  fact-checker, evaluator x2) uses `.with_structured_output()` against a
  real Pydantic schema (`llm/schemas.py`) - the model is constrained to
  that shape by LangChain, not asked nicely for JSON and parsed by hand.
  Verified: both the Anthropic and Ollama paths construct correctly and
  `.with_structured_output()` wires up without error (see commit history /
  build notes) - live call quality itself is unverified in this environment.
- **Real LangSmith**: two layers, not one.
  1. **Explicit, code-level tracing**: every agent function (`mode_router`,
     `fact_checker`, `evaluator` x2, `explainer`, `tech_quizzer`,
     `question_generator`, `followup_prober`, `report_synthesizer`) has a
     real `@traceable` decorator from the `langsmith` SDK - each shows up as
     a named span in a trace, not just raw model calls. This is present in
     the code regardless of whether tracing is enabled; it activates when
     `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set.
  2. **A real dataset + experiment**: `eval/langsmith_consistency_eval.py`
     uses the actual `langsmith` `Client` to create a real dataset from
     fixed test cases, then runs a real `evaluate(..., num_repetitions=N)`
     experiment - each repeated scoring run is a real, inspectable run in
     your LangSmith dashboard, not a local print statement. Requires a
     real LangSmith account; there's no mock fallback for this one, since
     the point is exercising the actual platform.
  Every call site (`Client()`, `create_dataset`, `create_examples`,
  `evaluate`) was checked against the real installed SDK's signatures
  during development - not guessed at. What's NOT verified: the exact
  output of a live run, since no LangSmith API key was available in the
  build environment. The script prints `results.to_pandas()` on first run
  specifically so you can see the real shape and extend from there.

## Honest scope notes

- A real (even small) user-interaction study was considered and deliberately
  not included in this version - a documented, reasoned scope decision, not
  an oversight.
- Starting a new quiz while a behavioral question is pending overwrites that
  pending question (it stays uncovered and gets re-asked later) - a known,
  accepted MVP limitation rather than a full dual-pending-slot design.
- All testing here was done with MockProvider and a scripted test provider
  for branch-logic verification. Real routing/scoring QUALITY with Ollama or
  Anthropic has not been verified in this environment - only the plumbing is
  proven. Run `--provider ollama` yourself to check real behavior.
