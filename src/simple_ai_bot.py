"""
simple_ai_bot.py — Single-turn AI chatbot via LangGraph.

Enhancements over the original:
  - Config loaded from .env via config.py
  - Streaming responses (token-by-token output)
  - Clean REPL loop with proper exit handling
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_NUM_PREDICT, OLLAMA_VALIDATE_ON_INIT


# ── LLM setup ─────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=OLLAMA_NUM_PREDICT,
)


# ── State definition ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    message: List[HumanMessage]


# ── Graph node ────────────────────────────────────────────────────────────
def process_message(state: AgentState) -> AgentState:
    """Stream the LLM response token-by-token."""
    print("\nAssistant: ", end="", flush=True)
    for chunk in llm.stream(state["message"]):
        print(chunk.content, end="", flush=True)
    print()  # newline after stream
    return state


# ── Build graph ───────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)


# ── Main REPL ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== SIMPLE AI BOT ===")
    print("Type 'exit' to quit.\n")

    app = graph.compile()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        app.invoke({"message": [HumanMessage(content=user_input)]})
