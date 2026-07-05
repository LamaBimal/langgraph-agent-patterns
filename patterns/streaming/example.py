"""
example.py — Streaming: Live token output + step progress

Demonstrates all four streaming modes on the same agent:
  1. Token mode   — characters appear as they are generated
  2. Steps mode   — which node is running and what changed
  3. Updates mode — delta state changes per node
  4. SSE mode     — formatted output for web delivery

Run:
    python patterns/streaming/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from streaming_pattern import print_tokens, print_steps, stream_updates, to_sse
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── LLM and tools ─────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '15 + 7 * 3'."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [calculate]
llm_with_tools = llm.bind_tools(tools)


# ── State and graph ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_llm(state: AgentState) -> AgentState:
    system = SystemMessage(content="You are a helpful assistant. Use the calculate tool for math.")
    response = llm_with_tools.invoke([system] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "llm")
app = graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== STREAMING PATTERNS EXAMPLE =====\n")

    question = input("Ask a question (or press Enter for default): ").strip()
    if not question:
        question = "What is (123 + 456) multiplied by 7? Explain your working."

    initial = {"messages": [HumanMessage(content=question)]}

    print("\n─── MODE 1: Token streaming ───")
    print_tokens(app, initial, prefix="Assistant: ")

    print("\n─── MODE 2: Step progress ───")
    print_steps(app, initial)

    print("\n─── MODE 3: Delta updates ───")
    for update in stream_updates(app, initial):
        node = update["node"]
        keys = list(update["updates"].keys()) if isinstance(update["updates"], dict) else []
        print(f"  [{node}] updated keys: {keys}")

    print("\n─── MODE 4: SSE format sample ───")
    count = 0
    for token in ["Hello", " world", "!"]:
        print(repr(to_sse({"token": token}, event_type="token")))
        count += 1
        if count >= 3:
            break
    print(repr(to_sse({"done": True}, event_type="done")))
