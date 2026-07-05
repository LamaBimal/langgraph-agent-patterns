"""
example.py — Checkpointing: Multi-step Research Agent

A 3-step research pipeline (search → analyse → summarise) with SQLite
checkpointing. Demonstrates:
  - Auto-saving state after each step
  - Resuming a workflow from the last checkpoint after interruption
  - Inspecting full checkpoint history
  - Time-travel: re-running from a prior step

Run:
    python patterns/checkpointing/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from checkpoint_pattern import CheckpointerFactory, CheckpointExplorer, thread_config
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)


# ── State ──────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_results: str
    analysis: str
    summary: str
    step: int


# ── Nodes ──────────────────────────────────────────────────────────────────
def search_step(state: ResearchState) -> ResearchState:
    """Step 1: Simulate searching for information on the topic."""
    print(f"\n[Step 1/3] Searching for: {state['topic']}")
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Provide key facts about the topic."),
        HumanMessage(content=f"Find key facts about: {state['topic']}")
    ])
    print(f"  → Found {len(response.content)} chars of results")
    return {
        "search_results": response.content,
        "step": 1,
        "messages": [AIMessage(content=f"Search complete: {response.content[:100]}...")]
    }


def analyse_step(state: ResearchState) -> ResearchState:
    """Step 2: Analyse the search results."""
    print(f"\n[Step 2/3] Analysing results...")
    response = llm.invoke([
        SystemMessage(content="Analyse the following research findings. Identify key themes and insights."),
        HumanMessage(content=state["search_results"])
    ])
    print(f"  → Analysis complete ({len(response.content)} chars)")
    return {
        "analysis": response.content,
        "step": 2,
        "messages": [AIMessage(content=f"Analysis: {response.content[:100]}...")]
    }


def summarise_step(state: ResearchState) -> ResearchState:
    """Step 3: Produce a concise executive summary."""
    print(f"\n[Step 3/3] Summarising...")
    response = llm.invoke([
        SystemMessage(content="Write a concise 3-sentence executive summary of the analysis."),
        HumanMessage(content=state["analysis"])
    ])
    print(f"  → Summary ready")
    return {
        "summary": response.content,
        "step": 3,
        "messages": [AIMessage(content=response.content)]
    }


# ── Graph ──────────────────────────────────────────────────────────────────
graph = StateGraph(ResearchState)
graph.add_node("search", search_step)
graph.add_node("analyse", analyse_step)
graph.add_node("summarise", summarise_step)

graph.add_edge(START, "search")
graph.add_edge("search", "analyse")
graph.add_edge("analyse", "summarise")
graph.add_edge("summarise", END)

# Use SQLite checkpointer — state persists across runs
checkpointer = CheckpointerFactory.sqlite("./research_checkpoints.db")
app = graph.compile(checkpointer=checkpointer)


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== CHECKPOINTING EXAMPLE =====")

    topic = input("Research topic (or press Enter for 'Large Language Models'): ").strip()
    if not topic:
        topic = "Large Language Models"

    session_id = f"research-{topic[:20].replace(' ', '-').lower()}"
    config = thread_config(session_id)
    explorer = CheckpointExplorer(app, config)

    # Check if we can resume an existing session
    existing = explorer.list_checkpoints()
    if existing:
        print(f"\n Found {len(existing)} existing checkpoints for '{topic}'")
        explorer.print_history()
        resume = input("\nResume from last checkpoint? [y/n]: ").strip().lower()
        if resume == "y":
            print("\nResuming workflow...")
            result = app.invoke(None, config=config)
        else:
            print("\nStarting fresh...")
            result = app.invoke(
                {"messages": [HumanMessage(content=topic)], "topic": topic,
                 "search_results": "", "analysis": "", "summary": "", "step": 0},
                config=config
            )
    else:
        result = app.invoke(
            {"messages": [HumanMessage(content=topic)], "topic": topic,
             "search_results": "", "analysis": "", "summary": "", "step": 0},
            config=config
        )

    print(f"\n{'='*55}")
    print("RESEARCH COMPLETE")
    print(f"{'='*55}")
    print(f"\n📋 SUMMARY:\n{result.get('summary', 'No summary generated')}")

    # Show checkpoint history after run
    explorer.print_history()
