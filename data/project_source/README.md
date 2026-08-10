# Research assistant orchestrator

A small multi-agent system: one orchestrator routes each message to one of
three specialist agents (or several, for compound requests), one of which
does real retrieval-augmented generation over a set of short papers about
orchestration, RAG, evaluation, tool use, and prompt engineering.

Built as interview prep for a role involving multi-agent orchestration and
RAG - see the "concepts" section below for how each piece maps to those
ideas.

## Setup

```bash
pip install -r requirements.txt
```

Three provider options, none of which require money:

- **mock** (default) - no setup, canned answers, real RAG/routing plumbing.
- **ollama** - fully free and local. If you already have Ollama with a chat
  model (e.g. `llama3.2:1b`) and an embedding model (e.g. `nomic-embed-text`),
  you're set - this mode uses both: the chat model for routing/answers, the
  embedding model for real semantic RAG (not the TF-IDF fallback). Otherwise
  install [Ollama](https://ollama.com) and run
  `ollama pull llama3.2:1b && ollama pull nomic-embed-text`.
- **anthropic** - real Claude-backed answers. New accounts get ~$5 in free
  trial credit (no card required to claim it), which comfortably covers
  everything in this project. Copy `.env.example` to `.env` and add your key
  if you want to use this.

## Running it

```bash
python -m src.main                                    # mock: instant, no setup
python -m src.main --provider ollama                   # free, local, real answers
python -m src.main --provider ollama --ollama-model llama3.2:3b
python -m src.main --provider anthropic                 # paid API, best answers
```

Try asking things like:
- "What is retrieval-augmented generation?" -> routes to `paper_qa`
- "Summarize the orchestration paper." -> routes to `summarizer`
- "Quiz me on that." (after a previous answer) -> routes to `quiz`
- "Summarize the RAG paper and then quiz me on it." -> routes to both, in order

## Running the evaluation

```bash
python -m eval.run_eval                          # mock provider
python -m eval.run_eval --provider ollama        # free, local, real routing
python -m eval.run_eval --provider anthropic     # paid API
```

Scores routing accuracy against `eval/eval_set.jsonl`, a labeled set of
single-turn and multi-turn cases. In mock mode you'll see one intentional
failure - explained below, it's informative, not a bug to fix.

## Running the tests

```bash
python -m pytest tests/
```

## Project layout

```
src/
  llm/            provider abstraction: mock, free local Ollama, or paid Anthropic
  rag/            embeddings, hand-rolled vector store, indexing
  agents/         paper_qa (RAG), summarizer, quiz
  orchestrator/   agent registry, planner, LangGraph graph
  data/           sample "papers" (original text, not real papers)
eval/             labeled eval set + scoring script
tests/            retrieval sanity tests
```

## Concepts, and where they live in the code

- **RAG**: `src/rag/` - chunking, embedding, vector search, all hand-rolled
  so the mechanism is visible rather than hidden in a library. `TfidfEmbedder`
  (word-overlap vectors, no model needed) is the fallback default;
  `OllamaEmbedder` (real neural embeddings via a local model like
  `nomic-embed-text`) is used automatically in `--provider ollama` mode -
  that's the difference between the simplified demo and actual semantic RAG.
- **Orchestration**: `src/orchestrator/graph.py` - a LangGraph `StateGraph`
  with three nodes (plan -> execute -> synthesize). State carries the
  conversation history across turns, which is how multi-turn memory works
  ("quiz me on that" resolves because the quiz agent reads recent history).
- **Agent descriptions/interfaces**: `src/orchestrator/registry.py` - each
  agent is a structured spec (name, description, examples), not a hardcoded
  if/else branch. The planner prompt is built from this registry, so adding
  a new agent doesn't require touching the routing logic.
- **Compound request handling**: the planner can return more than one step
  (see `planner.py`); `execute_node` runs them in order and `synthesize_node`
  merges multiple outputs into one reply.
- **Benchmarks/evaluation**: `eval/` - a labeled dataset plus a script that
  measures routing accuracy per turn, including multi-turn cases.

## About the one eval failure in mock mode

`MockProvider`'s "planner" is a naive keyword check (see `mock_provider.py`)
used only so the whole system can run without an API key. In the
`multi_turn_summary_then_question` case, the word "summarize" from an
*earlier* turn leaks into the routing decision for a later, unrelated
message, misrouting it. This is a genuine illustration of why real
orchestration needs language understanding rather than keyword matching -
run with `--live` and a real model resolves it correctly by actually
reading which parts of the prompt are the new request versus prior context.

## Notes

- `anthropic`, `langgraph` pull in `langsmith` as a dependency. Setting the
  standard `LANGCHAIN_TRACING_V2=true` / `LANGCHAIN_API_KEY` environment
  variables (with a free LangSmith account) will trace every planner and
  agent call automatically - useful for seeing exactly what context each
  agent received, without changing any code here.
- Forms are never auto-submitted, nothing here calls external services
  except the optional Anthropic API call in `--live` mode.
