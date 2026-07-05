"""
example.py — Human-in-the-Loop: Email Drafting Agent

The agent drafts an email based on user instructions. Before "sending"
(printing), it pauses for human approval. The human can:
  - approve  → email is sent
  - modify   → agent redrafts incorporating feedback
  - reject   → workflow ends without sending

Run:
    python patterns/human_in_the_loop/example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)


# ── State ──────────────────────────────────────────────────────────────────
class EmailState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft_email: str
    human_approved: bool
    human_feedback: str
    sent: bool


# ── Nodes ──────────────────────────────────────────────────────────────────
def draft_email(state: EmailState) -> EmailState:
    """LLM drafts an email based on conversation history."""
    feedback = state.get("human_feedback", "")
    system = SystemMessage(content=(
        "You are an email writing assistant. Draft a professional email "
        "based on the user's instructions. Output ONLY the email text, "
        "no explanations."
        + (f"\n\nRevision feedback: {feedback}" if feedback else "")
    ))
    response = llm.invoke([system] + list(state["messages"]))
    draft = response.content
    print(f"\n📧 DRAFT EMAIL:\n{'-'*40}\n{draft}\n{'-'*40}")
    return {"draft_email": draft, "messages": [AIMessage(content=draft)]}


def human_review(state: EmailState) -> EmailState:
    """
    INTERRUPT POINT — pauses here for human review.
    In a real system this node would notify via Slack/email/webhook.
    The graph is compiled with interrupt_before=["send_email"] so this
    node actually runs but execution stops before send_email.
    """
    print("\n⏸️  Waiting for human approval...")
    return {}


def send_email(state: EmailState) -> EmailState:
    """Simulate sending the email after approval."""
    print(f"\n✅ EMAIL SENT:\n{state['draft_email']}")
    return {"sent": True}


def end_rejected(state: EmailState) -> EmailState:
    print("\n❌ Email rejected. Workflow ended without sending.")
    return {"sent": False}


def route_after_review(state: EmailState) -> str:
    decision = state.get("human_approved")
    if decision is True:
        return "send"
    elif state.get("human_feedback"):
        return "redraft"
    return "reject"


# ── Graph ──────────────────────────────────────────────────────────────────
graph = StateGraph(EmailState)
graph.add_node("draft_email", draft_email)
graph.add_node("human_review", human_review)
graph.add_node("send_email", send_email)
graph.add_node("end_rejected", end_rejected)

graph.add_edge(START, "draft_email")
graph.add_edge("draft_email", "human_review")
graph.add_conditional_edges(
    "human_review",
    route_after_review,
    {"send": "send_email", "redraft": "draft_email", "reject": "end_rejected"},
)
graph.add_edge("send_email", END)
graph.add_edge("end_rejected", END)

checkpointer = MemorySaver()
# interrupt_before="human_review" means graph pauses BEFORE executing human_review
app = graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

thread = {"configurable": {"thread_id": "email-session-1"}}


# ── Runner ─────────────────────────────────────────────────────────────────
def run():
    print("\n===== HUMAN-IN-THE-LOOP: Email Agent =====")
    instructions = input("\nDescribe the email to write: ").strip()
    if not instructions:
        instructions = "Write a follow-up email to a client asking for project status update"

    initial_state: EmailState = {
        "messages": [HumanMessage(content=instructions)],
        "draft_email": "",
        "human_approved": False,
        "human_feedback": "",
        "sent": False,
    }

    # First run — stops before human_review
    app.invoke(initial_state, config=thread)

    while True:
        current = app.get_state(thread)
        draft = current.values.get("draft_email", "")
        print(f"\n📧 Current draft:\n{'-'*40}\n{draft}\n{'-'*40}")
        print("\nOptions: [approve] [reject] [modify <feedback>]")
        raw = input("Decision: ").strip()

        if raw.lower() == "approve":
            app.update_state(thread, {"human_approved": True, "human_feedback": ""})
            app.invoke(None, config=thread)
            break
        elif raw.lower() == "reject":
            app.update_state(thread, {"human_approved": False, "human_feedback": ""})
            app.invoke(None, config=thread)
            break
        elif raw.lower().startswith("modify"):
            feedback = raw[6:].strip() or input("Enter feedback: ").strip()
            app.update_state(thread, {
                "human_approved": False,
                "human_feedback": feedback,
                "messages": [HumanMessage(content=f"Revision: {feedback}")]
            })
            # Resume — goes back to draft_email with feedback
            app.invoke(None, config=thread)
            # Will pause again before human_review — loop continues
        else:
            print("Unrecognised input. Try 'approve', 'reject', or 'modify <feedback>'")

    print("\n===== DONE =====")


if __name__ == "__main__":
    run()
