"""
example.py — Fallback Models: 3-tier Ollama model chain

Tries models in this order:
  1. deepseek-r1:8b  (primary — assume available locally)
  2. llama3.1:latest (secondary)
  3. Static message  (final fallback if both unavailable)

Each model is tested with a quality gate that rejects empty responses.

Run:
    python patterns/fallback_models/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from fallback_pattern import FallbackChain, QualityGate, not_empty, not_refusal, build_fallback_node
from config import OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── Models in priority order ───────────────────────────────────────────────
# Model 1: primary (fast, cost-effective)
primary_model = ChatOllama(
    model="deepseek-r1:8b",
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

# Model 2: secondary fallback
secondary_model = ChatOllama(
    model="llama3.1:latest",
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

# Quality gate: response must be non-empty and not a refusal
quality_gate = QualityGate(
    checks=[not_empty, not_refusal],
    min_length=20,
)

# Build the chain
chain = FallbackChain(
    models=[primary_model, secondary_model],
    quality_gate=quality_gate,
    static_fallback="I'm sorry, no models are currently available. Please try again later.",
)


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_used: str


# ── Graph ──────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("llm", build_fallback_node(chain))
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
app = graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== FALLBACK MODELS EXAMPLE =====")
    print("Chain: deepseek-r1:8b → llama3.1 → static fallback\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "model_used": "",
        })

        print(f"\nAssistant ({result['model_used']}): {result['messages'][-1].content}\n")
