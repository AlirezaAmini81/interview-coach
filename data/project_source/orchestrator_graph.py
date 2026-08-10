"""
The orchestrator, built as a LangGraph StateGraph with three nodes:

    plan -> execute -> synthesize

- plan: calls the planner to decide which agent(s) handle the new message
- execute: runs each planned step against the right agent, in order
- synthesize: combines step outputs into one reply (verbatim if there was
  only one step, LLM-merged if there were several)

State is a plain dict that flows through every node and accumulates the
conversation history across turns - that's the multi-turn memory: each call
to graph.invoke() passes in the growing history from the previous turns.
"""
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..agents import general_chat_agent, paper_qa_agent, quiz_agent, summarizer_agent
from ..llm.provider import LLMProvider
from ..rag.embeddings import Embedder
from ..rag.vector_store import VectorStore
from . import planner


class OrchestratorState(TypedDict):
    history: list[dict]
    user_message: str
    plan: list[dict]
    step_results: list[dict]
    response: str


def build_graph(llm: LLMProvider, vector_store: VectorStore, embedder: Embedder):
    def plan_node(state: OrchestratorState) -> dict[str, Any]:
        steps = planner.create_plan(state["history"], state["user_message"], llm)
        return {"plan": steps}

    def execute_node(state: OrchestratorState) -> dict[str, Any]:
        results = []
        for step in state["plan"]:
            agent_name, instruction = step["agent"], step["instruction"]

            if agent_name == "paper_qa":
                output = paper_qa_agent.run(instruction, vector_store, embedder, llm)
            elif agent_name == "summarizer":
                output = summarizer_agent.run(instruction, state["history"], vector_store, embedder, llm)
            elif agent_name == "quiz":
                output = quiz_agent.run(instruction, state["history"], llm)
            elif agent_name == "general_chat":
                output = general_chat_agent.run(instruction, state["history"], llm)
            else:
                output = f"(no such agent: {agent_name})"

            results.append({"agent": agent_name, "output": output})
        return {"step_results": results}

    def synthesize_node(state: OrchestratorState) -> dict[str, Any]:
        results = state["step_results"]
        if len(results) == 1:
            return {"response": results[0]["output"]}

        # More than one agent contributed - stitch outputs together with
        # clear headers rather than silently picking one.
        parts = [f"[{r['agent']}]\n{r['output']}" for r in results]
        return {"response": "\n\n".join(parts)}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
