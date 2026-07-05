"""
drafter_agent.py — Interactive document writing and saving agent.

Enhancements over the original:
  - Config loaded from .env via config.py
  - Streaming output for AI responses
  - document_content managed in AgentState (no global variable)
  - Cleaner graph: agent only advances to tools when tool calls exist
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_NUM_PREDICT, OLLAMA_VALIDATE_ON_INIT


# ── State definition ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_content: str          # current document managed in state, not a global


# ── Tools ─────────────────────────────────────────────────────────────────
@tool
def update(content: str) -> str:
    """Replace the entire document with new content provided by the user."""
    return f"DOCUMENT_UPDATE::{content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a .txt file and finish the session.

    Args:
        filename: Name of the output file (extension added automatically).
    """
    return f"DOCUMENT_SAVE::{filename}"


tools = [update, save]

# ── LLM setup ─────────────────────────────────────────────────────────────
model = ChatOllama(
    model=OLLAMA_MODEL,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=OLLAMA_NUM_PREDICT,
).bind_tools(tools)


# ── Graph nodes ───────────────────────────────────────────────────────────
def our_agent(state: AgentState) -> AgentState:
    """Prompt the user, call the LLM, and stream the response."""
    doc = state.get("document_content", "")

    system_prompt = SystemMessage(content=f"""
You are Drafter, a helpful writing assistant. Help the user create and modify documents.

- To update the document, call the 'update' tool with the complete new content.
- To save and finish, call the 'save' tool with a filename.
- Always show the current document state after any modification.

Current document content:
{doc if doc else "(empty — nothing written yet)"}
""")

    if not state["messages"]:
        user_input = "I'm ready to help you create a document. What would you like to write?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ").strip()
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # Stream the response
    print("\n🤖 AI: ", end="", flush=True)
    full_response = ""
    response = None
    for chunk in model.stream(all_messages):
        token = chunk.content
        print(token, end="", flush=True)
        full_response += token
        response = chunk   # keep last chunk (carries tool_calls metadata)
    print()

    # Re-invoke without streaming to get the full AIMessage with tool_calls intact
    response = model.invoke(all_messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {
        "messages": list(state["messages"]) + [user_message, response],
        "document_content": doc,
    }


def process_tool_results(state: AgentState) -> AgentState:
    """Execute tool calls and apply side-effects (update / save) to state."""
    doc = state.get("document_content", "")
    last_ai = state["messages"][-1]
    tool_results = []

    for tc in last_ai.tool_calls:
        name = tc["name"]
        args = tc["args"]

        if name == "update":
            doc = args.get("content", doc)
            result_content = (
                f"Document updated successfully.\n\nCurrent content:\n{doc}"
            )
            print(f"\n🛠️  TOOL [update]: document updated ({len(doc)} chars)")

        elif name == "save":
            filename: str = args.get("filename", "document")
            if not filename.endswith(".txt"):
                filename += ".txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(doc)
                result_content = f"Document saved to '{filename}'."
                print(f"\n🛠️  TOOL [save]: saved to '{filename}'")
            except Exception as e:
                result_content = f"Error saving document: {e}"
                print(f"\n🛠️  TOOL [save]: ERROR — {e}")
        else:
            result_content = f"Unknown tool: {name}"

        tool_results.append(
            ToolMessage(tool_call_id=tc["id"], name=name, content=result_content)
        )

    return {
        "messages": list(state["messages"]) + tool_results,
        "document_content": doc,
    }


# ── Routing logic ─────────────────────────────────────────────────────────
def agent_router(state: AgentState) -> str:
    """Go to tools if the model made tool calls, otherwise loop back to agent."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "agent"


def tool_router(state: AgentState) -> str:
    """End the session after a save, otherwise return to agent."""
    for message in reversed(state["messages"]):
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"
    return "continue"


# ── Build graph ───────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", process_tool_results)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", agent_router, {"tools": "tools", "agent": "agent"})
graph.add_conditional_edges("tools", tool_router, {"continue": "agent", "end": END})

app = graph.compile()


# ── Runner ────────────────────────────────────────────────────────────────
def run_document_agent() -> None:
    print("\n===== DRAFTER =====")
    print("Tell me what to write. Say 'save' when you're done.\n")

    state: AgentState = {"messages": [], "document_content": ""}

    for step in app.stream(state, stream_mode="values"):
        pass  # streaming output is printed inside nodes

    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
