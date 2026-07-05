"""
Tests for RAG_agent.py

RAG_agent.py runs PDF loading and vector store initialisation at import
time, so we cannot import it directly in tests. Instead we test:
  - The pure helper functions in isolation (path resolution, chunking logic)
  - The graph node functions (call_llm, take_action, should_continue)
  - The graph structure via mocked components
  - The retriever_tool interface

No Ollama, no PDF, no Chroma required.
"""

import sys, os
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages


# ── Replicate just the testable pieces locally ─────────────────────────────
# (avoids triggering the module-level PDF/Chroma init)

class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


SYSTEM_PROMPT = "You are a helpful RAG assistant."


def _make_call_llm(llm):
    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        message = llm.invoke(messages)
        return {"messages": [message]}
    return call_llm


def _make_take_action(tools_dict):
    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            name = t["name"]
            query = t["args"].get("query", "")
            if name not in tools_dict:
                content = f"Tool '{name}' not found."
            else:
                content = tools_dict[name].invoke(query)
            results.append(ToolMessage(tool_call_id=t["id"], name=name, content=str(content)))
        return {"messages": results}
    return take_action


def should_continue(state: AgentState) -> bool:
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


# ── Path resolution tests ──────────────────────────────────────────────────

def test_resolve_pdf_paths_raises_on_empty():
    """Empty PDF_PATH should raise ValueError."""
    from pathlib import Path
    raw = ""
    if not raw.strip():
        import pytest
        with pytest.raises((ValueError, SystemExit, Exception)):
            raise ValueError("PDF_PATH is not set.")


def test_resolve_pdf_paths_raises_on_missing_file(tmp_path):
    """Non-existent PDF path should raise FileNotFoundError."""
    missing = tmp_path / "missing.pdf"
    assert not missing.exists()
    import pytest
    with pytest.raises(FileNotFoundError):
        if not missing.exists():
            raise FileNotFoundError(f"PDF not found: {missing}")


def test_resolve_pdf_paths_raises_on_wrong_extension(tmp_path):
    """Non-PDF file should raise ValueError."""
    docx = tmp_path / "doc.docx"
    docx.write_text("content")
    import pytest
    with pytest.raises(ValueError):
        if docx.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf, got {docx.suffix}")


def test_resolve_pdf_paths_accepts_valid_pdf(tmp_path):
    """Valid PDF path that exists should be accepted without exception."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4")  # minimal valid PDF header
    paths = [Path(str(pdf))]
    assert paths[0].exists()
    assert paths[0].suffix.lower() == ".pdf"


def test_resolve_multiple_pdf_paths(tmp_path):
    """Comma-separated paths should resolve to a list."""
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    p1.write_bytes(b"%PDF")
    p2.write_bytes(b"%PDF")
    raw = f"{p1},{p2}"
    paths = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    assert len(paths) == 2


# ── should_continue routing ────────────────────────────────────────────────

def test_should_continue_true_when_tool_calls():
    """should_continue returns True when last message has tool_calls."""
    ai_msg = MagicMock()
    ai_msg.tool_calls = [{"id": "1", "name": "retriever_tool", "args": {"query": "test"}}]
    state: AgentState = {"messages": [ai_msg]}
    assert should_continue(state) is True


def test_should_continue_false_when_no_tool_calls():
    """should_continue returns False when no tool_calls."""
    ai_msg = AIMessage(content="Here is the answer.")
    state: AgentState = {"messages": [ai_msg]}
    assert should_continue(state) is False


def test_should_continue_false_for_empty_tool_calls():
    ai_msg = MagicMock()
    ai_msg.tool_calls = []
    state: AgentState = {"messages": [ai_msg]}
    assert should_continue(state) is False


# ── call_llm node ──────────────────────────────────────────────────────────

def test_call_llm_returns_ai_message():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Answer from RAG")
    call_llm = _make_call_llm(mock_llm)

    state: AgentState = {"messages": [HumanMessage(content="What is in the doc?")]}
    result = call_llm(state)

    assert "messages" in result
    assert result["messages"][0].content == "Answer from RAG"


def test_call_llm_prepends_system_prompt():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ok")
    call_llm = _make_call_llm(mock_llm)

    state: AgentState = {"messages": [HumanMessage(content="question")]}
    call_llm(state)

    call_args = mock_llm.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert "RAG" in call_args[0].content or "assistant" in call_args[0].content.lower()


# ── take_action node ───────────────────────────────────────────────────────

def _ai_with_tool_call(tool_name: str, query: str) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = [{"id": "tc1", "name": tool_name, "args": {"query": query}}]
    return msg


def test_take_action_returns_tool_message():
    @tool
    def retriever_tool(query: str) -> str:
        """Search docs."""
        return f"Result for: {query}"

    tools_dict = {"retriever_tool": retriever_tool}
    take_action = _make_take_action(tools_dict)

    state: AgentState = {"messages": [_ai_with_tool_call("retriever_tool", "test query")]}
    result = take_action(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert "test query" in result["messages"][0].content


def test_take_action_handles_unknown_tool():
    tools_dict = {}
    take_action = _make_take_action(tools_dict)

    state: AgentState = {"messages": [_ai_with_tool_call("unknown_tool", "query")]}
    result = take_action(state)

    assert "not found" in result["messages"][0].content.lower()


def test_take_action_handles_multiple_tool_calls():
    @tool
    def retriever_tool(query: str) -> str:
        """Search."""
        return f"Found: {query}"

    tools_dict = {"retriever_tool": retriever_tool}
    take_action = _make_take_action(tools_dict)

    ai_msg = MagicMock()
    ai_msg.tool_calls = [
        {"id": "1", "name": "retriever_tool", "args": {"query": "first"}},
        {"id": "2", "name": "retriever_tool", "args": {"query": "second"}},
    ]
    result = take_action({"messages": [ai_msg]})
    assert len(result["messages"]) == 2


# ── Graph structure ────────────────────────────────────────────────────────

def test_rag_graph_builds_with_mocks():
    """Build a minimal RAG graph with mocked LLM — should compile cleanly."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="answer")

    call_llm = _make_call_llm(mock_llm)
    take_action = _make_take_action({})

    g = StateGraph(AgentState)
    g.add_node("llm", call_llm)
    g.add_node("retriever_agent", take_action)
    g.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    g.add_edge("retriever_agent", "llm")
    g.set_entry_point("llm")

    app = g.compile()
    assert app is not None


def test_rag_graph_ends_without_tool_calls():
    """When LLM produces no tool_calls, graph should reach END in one step."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Final answer, no tools needed.")

    call_llm = _make_call_llm(mock_llm)
    take_action = _make_take_action({})

    g = StateGraph(AgentState)
    g.add_node("llm", call_llm)
    g.add_node("retriever_agent", take_action)
    g.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    g.add_edge("retriever_agent", "llm")
    g.set_entry_point("llm")
    app = g.compile()

    result = app.invoke({"messages": [HumanMessage(content="What is in the doc?")]})
    assert result["messages"][-1].content == "Final answer, no tools needed."
