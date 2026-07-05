"""
ReAct_agent.py — Reasoning + Acting agent with math and web search tools.

Enhancements over the original:
  - Config loaded from .env via config.py
  - Web search tool added via DuckDuckGo (no API key required)
  - Streaming responses via agent.stream()
  - Interactive REPL loop instead of a single hardcoded input
  - Cleaned up commented-out dead code
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_NUM_PREDICT, OLLAMA_VALIDATE_ON_INIT


# ── State definition ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Math tools ────────────────────────────────────────────────────────────
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print(f"  [tool] add({a}, {b})")
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    print(f"  [tool] subtract({a}, {b})")
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"  [tool] multiply({a}, {b})")
    return a * b


# ── Web search tool ───────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return a summary of results.
    Use this for current events, facts, or anything beyond training data.
    """
    print(f"  [tool] search_web('{query}')")
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        _search = DuckDuckGoSearchRun()
        return _search.run(query)
    except ImportError:
        return "Web search is unavailable. Install duckduckgo-search: pip install duckduckgo-search"


tools = [add, subtract, multiply, search_web]

# ── LLM setup ─────────────────────────────────────────────────────────────
model = ChatOllama(
    model=OLLAMA_MODEL,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=OLLAMA_NUM_PREDICT,
).bind_tools(tools)


# ── Graph nodes ───────────────────────────────────────────────────────────
def model_call(state: AgentState) -> AgentState:
    """Call the LLM with the current message history."""
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. Use tools when needed to give accurate answers."
    )
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools if the model made tool calls, otherwise end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"


# ── Build graph ───────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {"end": END, "continue": "tools"},
)
graph.add_edge("tools", "our_agent")

agent = graph.compile()


# ── Streaming helper ──────────────────────────────────────────────────────
def stream_response(user_input: str) -> None:
    """Stream the agent's response for a given user input."""
    state = {"messages": [HumanMessage(content=user_input)]}
    print()
    for step in agent.stream(state, stream_mode="values"):
        last = step["messages"][-1]
        last.pretty_print()


# ── Main REPL ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== ReAct AGENT ===")
    print("Tools available: add, subtract, multiply, search_web")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        stream_response(user_input)
