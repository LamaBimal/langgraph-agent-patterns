"""
Tests for drafter_agent.py — tool logic and graph structure.

Tool functions return sentinel strings that process_tool_results
interprets; we test the tool return values and the graph structure.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drafter_agent import update, save, app, tool_router, AgentState
from langchain_core.messages import ToolMessage


# ── Tool return value tests ────────────────────────────────────────────────

def test_update_returns_sentinel():
    """update tool should return a string starting with DOCUMENT_UPDATE::"""
    result = update.invoke({"content": "Hello World"})
    assert result.startswith("DOCUMENT_UPDATE::")
    assert "Hello World" in result


def test_update_empty_content():
    """update tool handles empty string."""
    result = update.invoke({"content": ""})
    assert "DOCUMENT_UPDATE::" in result


def test_save_returns_sentinel():
    """save tool should return a string starting with DOCUMENT_SAVE::"""
    result = save.invoke({"filename": "test_output"})
    assert result.startswith("DOCUMENT_SAVE::")
    assert "test_output" in result


def test_save_adds_txt_extension():
    """save tool result should include the filename."""
    result = save.invoke({"filename": "my_doc"})
    assert "my_doc" in result


# ── tool_router logic tests ────────────────────────────────────────────────

def _make_state_with_tool_message(content: str) -> AgentState:
    """Helper to build a minimal AgentState with a single ToolMessage."""
    return {
        "messages": [ToolMessage(tool_call_id="x", name="save", content=content)],
        "document_content": "",
    }


def test_tool_router_ends_on_save():
    """tool_router should return 'end' when a save ToolMessage is present."""
    state = _make_state_with_tool_message("Document saved to output.txt.")
    assert tool_router(state) == "end"


def test_tool_router_continues_on_update():
    """tool_router should return 'continue' for non-save messages."""
    state = _make_state_with_tool_message("Document updated successfully.")
    assert tool_router(state) == "continue"


def test_tool_router_continues_empty():
    """tool_router should return 'continue' when there are no messages."""
    state: AgentState = {"messages": [], "document_content": ""}
    assert tool_router(state) == "continue"


# ── Graph structure test ───────────────────────────────────────────────────

def test_app_compiled():
    """Drafter graph should compile without error."""
    assert app is not None


def test_app_has_nodes():
    """Drafter graph should have 'agent' and 'tools' nodes."""
    node_names = list(app.get_graph().nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names
