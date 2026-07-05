"""
pattern.py — Human-in-the-Loop reusable building block.

Provides a HumanApprovalGate that can be dropped into any LangGraph
graph to pause execution and wait for human review before continuing.

Usage:
    from patterns.human_in_the_loop.pattern import (
        build_approval_graph, ApprovalState, HumanDecision
    )
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ── State ──────────────────────────────────────────────────────────────────

class ApprovalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_action: str          # action the agent wants to take
    human_decision: str          # "approve" | "reject" | "modify"
    human_feedback: str          # optional feedback from human
    action_result: str           # result after execution


# ── Decision type ──────────────────────────────────────────────────────────

HumanDecision = Literal["approve", "reject", "modify"]


# ── Reusable nodes ─────────────────────────────────────────────────────────

def check_human_decision(state: ApprovalState) -> Literal["execute", "replan", "end"]:
    """
    Router: decide what happens after human reviews the pending action.
    - approve  → execute the action
    - modify   → replan with the human's feedback incorporated
    - reject   → end the workflow
    """
    decision = state.get("human_decision", "").lower()
    if decision == "approve":
        return "execute"
    elif decision == "modify":
        return "replan"
    else:
        return "end"


def format_pending_action(action: str, feedback: str = "") -> str:
    """
    Format a clear human-readable summary of what the agent wants to do.
    Optionally includes prior feedback for re-plans.
    """
    lines = [
        "=" * 50,
        "PENDING ACTION — HUMAN REVIEW REQUIRED",
        "=" * 50,
        action,
    ]
    if feedback:
        lines += ["", f"Previous feedback: {feedback}"]
    lines += ["=" * 50]
    return "\n".join(lines)


def request_human_input(state: ApprovalState) -> dict:
    """
    Display the pending action and collect human decision interactively.
    In production, replace this with an API call, Slack message, etc.
    """
    print("\n" + format_pending_action(
        state["pending_action"],
        state.get("human_feedback", "")
    ))
    print("\nOptions: [approve] [reject] [modify]")

    decision = input("Your decision: ").strip().lower()
    while decision not in ("approve", "reject", "modify"):
        decision = input("Please enter 'approve', 'reject', or 'modify': ").strip().lower()

    feedback = ""
    if decision in ("modify", "reject"):
        feedback = input("Feedback (optional): ").strip()

    return {
        "human_decision": decision,
        "human_feedback": feedback,
    }


# ── Graph builder ──────────────────────────────────────────────────────────

def build_approval_graph(
    agent_node,
    execute_node,
    replan_node=None,
    interrupt_before: list[str] = None,
) -> tuple:
    """
    Build a graph with a human approval gate wired in.

    Args:
        agent_node:    function(state) → state  — drafts the pending_action
        execute_node:  function(state) → state  — executes an approved action
        replan_node:   function(state) → state  — re-plans after modification
                       (defaults to agent_node if not provided)
        interrupt_before: nodes to interrupt before (default: ["human_review"])

    Returns:
        (compiled_graph, checkpointer)
    """
    if replan_node is None:
        replan_node = agent_node
    if interrupt_before is None:
        interrupt_before = ["human_review"]

    graph = StateGraph(ApprovalState)
    graph.add_node("agent", agent_node)
    graph.add_node("human_review", request_human_input)
    graph.add_node("execute", execute_node)
    graph.add_node("replan", replan_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "human_review")
    graph.add_conditional_edges(
        "human_review",
        check_human_decision,
        {"execute": "execute", "replan": "replan", "end": END},
    )
    graph.add_edge("execute", END)
    graph.add_edge("replan", "human_review")

    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )
    return compiled, checkpointer
