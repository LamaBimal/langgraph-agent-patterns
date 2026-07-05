"""
example.py — Observability: Traced ReAct Agent

Every node execution is automatically timed and logged.
After the agent finishes, a full execution timeline is printed
with per-node latency and estimated token usage.

Run:
    python patterns/observability/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from observability_pattern import AgentTracer, traced_node, setup_logging
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT
import uuid

setup_logging(json_format=False)  # Human-readable for demo

# ── LLM and tools ─────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str


# ── Tracer ─────────────────────────────────────────────────────────────────
tracer = AgentTracer(session_id=str(uuid.uuid4())[:8])


# ── Nodes with tracing ─────────────────────────────────────────────────────
@traced_node(tracer, "llm_call")
def call_llm(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are a helpful math assistant. Use tools when needed.")
    response = llm_with_tools.invoke([system] + list(state["messages"]))
    return {"messages": [response]}


@traced_node(tracer, "tool_execution")
def run_tools(state: AgentState) -> AgentState:
    tool_node = ToolNode(tools)
    return tool_node.invoke(state)


# ── Graph ──────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", run_tools)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "llm")
app = graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== OBSERVABILITY EXAMPLE =====\n")

    question = input("Math question (or press Enter for default): ").strip()
    if not question:
        question = "What is (15 + 7) multiplied by 3?"

    result = app.invoke({
        "messages": [HumanMessage(content=question)],
        "session_id": tracer.session_id,
    })

    print(f"\n📊 ANSWER: {result['messages'][-1].content}")
    tracer.print_timeline()
