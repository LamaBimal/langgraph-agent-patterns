"""
example.py — Retry and Error Recovery: Flaky Tool Agent

Simulates an agent calling a flaky external API tool that fails
randomly. The graph retries up to 3 times with exponential backoff.
After exhausting retries, it routes to an error handler that returns
a safe fallback response.

Run:
    python patterns/retry_recovery/example.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from retry_pattern import RetryConfig, with_retry, should_retry, build_error_handler
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT

# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

# ── Retry policy ──────────────────────────────────────────────────────────
retry_config = RetryConfig(
    max_attempts=3,
    base_delay=0.5,    # short for demo; use 2-5s in production
    max_delay=10.0,
    jitter=True,
)

# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    attempt: int
    last_error: str
    error_history: list
    result: str


# ── Simulated flaky tool ───────────────────────────────────────────────────
_call_count = 0

def flaky_weather_api(city: str) -> str:
    """Simulates an API that fails on first 2 calls, succeeds on 3rd."""
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError(f"API timeout (simulated failure #{_call_count})")
    return f"Weather in {city}: 22°C, partly cloudy"


# ── Nodes ──────────────────────────────────────────────────────────────────
retry_cfg = RetryConfig(max_attempts=3, base_delay=0.3, jitter=False)

@with_retry(retry_cfg)
def call_weather_tool(state: AgentState) -> AgentState:
    """Call the flaky weather API with retry wrapping."""
    last_msg = state["messages"][-1]
    city = "London"  # In real agent, extract from LLM tool call
    print(f"\n  [tool] Calling weather API for '{city}' "
          f"(attempt {state.get('attempt', 0) + 1}/{retry_cfg.max_attempts})...")

    result = flaky_weather_api(city)
    print(f"  [tool] Success: {result}")
    return {"result": result, "messages": [AIMessage(content=result)]}


def generate_response(state: AgentState) -> AgentState:
    """LLM formulates final answer from tool result."""
    result = state.get("result", "")
    if result:
        response = llm.invoke([
            *state["messages"],
            HumanMessage(content=f"Tool result: {result}. Summarise for the user.")
        ])
        return {"messages": [AIMessage(content=response.content)]}
    return {}


# Error handler — returns a polite fallback when all retries fail
handle_error = build_error_handler(
    fallback_value="I'm sorry, the weather service is temporarily unavailable. Please try again later.",
    notify=lambda history: print(f"\n🚨 ALERT: {len(history)} failures — would notify on-call here"),
)


# ── Graph ──────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("call_tool", call_weather_tool)
graph.add_node("generate_response", generate_response)
graph.add_node("error_handler", handle_error)

graph.add_edge(START, "call_tool")
graph.add_conditional_edges(
    "call_tool",
    lambda s: "success" if not s.get("last_error") else should_retry(s, retry_cfg),
    {"success": "generate_response", "retry": "call_tool", "error_handler": "error_handler"},
)
graph.add_edge("generate_response", END)
graph.add_edge("error_handler", END)

app = graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== RETRY & ERROR RECOVERY EXAMPLE =====")
    print("Simulating a flaky API (fails twice, succeeds on 3rd call)\n")

    initial: AgentState = {
        "messages": [HumanMessage(content="What's the weather in London?")],
        "attempt": 0,
        "last_error": "",
        "error_history": [],
        "result": "",
    }

    result = app.invoke(initial)

    print("\n===== FINAL ANSWER =====")
    print(result["messages"][-1].content)
