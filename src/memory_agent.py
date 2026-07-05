"""
memory_agent.py — Multi-turn conversational agent with persistent memory.

Enhancements over the original:
  - Config loaded from .env via config.py (model, temperature, num_predict)
  - LangGraph MemorySaver checkpointer for cross-session memory persistence
  - Streaming responses (token-by-token output)
  - Conversation history auto-saved on exit
"""

from typing import TypedDict, Union, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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
    messages: List[Union[HumanMessage, AIMessage]]


# ── Graph node ────────────────────────────────────────────────────────────
def process_message(state: AgentState) -> AgentState:
    """Call the LLM and stream the response token-by-token."""
    print("\n AI: ", end="", flush=True)

    full_response = ""
    # Stream tokens as they arrive
    for chunk in llm.stream(state["messages"]):
        token = chunk.content
        print(token, end="", flush=True)
        full_response += token

    print()  # newline after stream completes

    state["messages"].append(AIMessage(content=full_response))
    return state


# ── Build graph with MemorySaver checkpointer ─────────────────────────────
memory = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compiling with checkpointer enables persistent memory across invocations
agent = graph.compile(checkpointer=memory)


# ── Main loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MEMORY AGENT ===")
    print("Conversation is persisted across turns via MemorySaver.")
    print("Type 'exit' to quit and save conversation history.\n")

    # thread_id scopes the memory — change it to start a fresh conversation
    config = {"configurable": {"thread_id": "session-1"}}

    conversation_history: List[Union[HumanMessage, AIMessage]] = []

    user_input = input("You: ")

    while user_input.lower() != "exit":
        conversation_history.append(HumanMessage(content=user_input))
        initial_state = {"messages": conversation_history}

        result = agent.invoke(initial_state, config=config)
        conversation_history = result["messages"]

        user_input = input("\nYou: ")

    # Save conversation to file on exit
    print("\nSaving the conversation history...")
    with open("conversation_history.txt", "w", encoding="utf-8") as f:
        for msg in conversation_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            f.write(f"{role}: {msg.content}\n\n")
    print("Conversation history saved to conversation_history.txt")
