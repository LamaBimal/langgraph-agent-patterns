"""
example.py — Memory: Personal Assistant with 3-tier Memory

Demonstrates:
  - Short-term: sliding window keeps last 10 messages
  - Long-term: facts extracted from conversation (name, preferences)
  - Episodic: session summary saved on exit, recalled next session

Run:
    python patterns/memory/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from memory_pattern import MemoryManager
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT
import datetime


# ── Setup ──────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

mem = MemoryManager(
    window_size=10,
    lt_storage="./patterns/memory/long_term_memory.json",
    ep_storage="./patterns/memory/episodic_memory.json",
)


# ── State ──────────────────────────────────────────────────────────────────
class AssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str


# ── Nodes ──────────────────────────────────────────────────────────────────
def respond(state: AssistantState) -> AssistantState:
    """Respond using LLM with memory context injected into system prompt."""
    # Build context from long-term and episodic memory
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = str(m.content)
            break

    context = mem.build_context_prompt(last_user_msg)

    system_content = "You are a helpful personal assistant with memory of past conversations."
    if context:
        system_content += f"\n\n{context}"

    # Extract facts from user message
    if last_user_msg:
        mem.extract_and_store_facts(last_user_msg)

    # Trim to window size
    trimmed = mem.short_term.trim(list(state["messages"]))

    response = llm.invoke([SystemMessage(content=system_content)] + trimmed)
    print(f"\nAssistant: {response.content}")
    return {"messages": [AIMessage(content=response.content)]}


# ── Graph ──────────────────────────────────────────────────────────────────
graph = StateGraph(AssistantState)
graph.add_node("respond", respond)
graph.add_edge(START, "respond")
graph.add_edge("respond", END)
app = graph.compile(checkpointer=MemorySaver())


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== MEMORY PATTERN: Personal Assistant =====")

    # Show any recalled facts from previous sessions
    all_facts = mem.long_term.all_facts()
    if all_facts:
        print(f"\n📚 Remembered from previous sessions:")
        for k, v in all_facts.items():
            print(f"  {k}: {v}")

    recent_eps = mem.episodic.recent(2)
    if recent_eps:
        print(f"\n🗂️  Recent episodes:")
        for ep in recent_eps:
            print(f"  [{ep['timestamp'][:10]}] {ep['summary'][:80]}...")

    session_id = f"session-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    messages = []
    print("\nType 'exit' to save this session and quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": messages, "session_id": session_id}, config=config)
        messages = list(result["messages"])

    # Save session as an episode
    if messages:
        conversation_text = " | ".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {str(m.content)[:50]}"
            for m in messages[-6:]
        )
        summary_response = llm.invoke([
            SystemMessage(content="Summarise this conversation in one sentence."),
            HumanMessage(content=conversation_text)
        ])
        mem.episodic.save_episode(
            session_id=session_id,
            summary=summary_response.content,
            topics=[w for w in conversation_text.split() if len(w) > 5][:5],
        )
        print(f"\n💾 Session saved: {summary_response.content}")

    print("\n===== SESSION ENDED =====")
