"""
Tests for simple_ai_bot.py

Verifies graph structure, state pass-through, streaming behaviour,
and edge cases. No Ollama required — LLM is mocked.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from simple_ai_bot import graph, AgentState, process_message


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm(tokens=("Hello", " there")):
    mock = MagicMock()
    mock.stream.return_value = iter([MagicMock(content=t) for t in tokens])
    return mock


# ── Graph structure ─────────────────────────────────────────────────────────

def test_graph_has_process_node():
    node_names = list(graph.compile().get_graph().nodes.keys())
    assert "process" in node_names


def test_graph_has_start_and_end():
    nodes = list(graph.compile().get_graph().nodes.keys())
    assert "__start__" in nodes
    assert "__end__" in nodes


def test_graph_compiles():
    assert graph.compile() is not None


# ── process_message node ───────────────────────────────────────────────────

def test_process_message_returns_state_unchanged():
    """process_message should return the same state (single-turn, no history)."""
    with patch("simple_ai_bot.llm", _mock_llm(("Hi",))):
        state: AgentState = {"message": [HumanMessage(content="Hello")]}
        result = process_message(state)
        # State is passed through unchanged — simple bot does not mutate it
        assert "message" in result
        assert result["message"][0].content == "Hello"


def test_process_message_calls_stream():
    """LLM .stream() must be called with the message list."""
    mock_llm = _mock_llm(("Hi",))
    with patch("simple_ai_bot.llm", mock_llm):
        state: AgentState = {"message": [HumanMessage(content="test")]}
        process_message(state)
        mock_llm.stream.assert_called_once()


def test_process_message_passes_messages_to_stream():
    """stream() should receive the message list from state."""
    mock_llm = _mock_llm(("Hi",))
    with patch("simple_ai_bot.llm", mock_llm):
        msgs = [HumanMessage(content="What is AI?")]
        process_message({"message": msgs})
        call_args = mock_llm.stream.call_args[0][0]
        assert call_args[0].content == "What is AI?"


# ── Full graph invocation ──────────────────────────────────────────────────

def test_invoke_returns_state():
    with patch("simple_ai_bot.llm", _mock_llm()):
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="Hi")]})
        assert "message" in result


def test_invoke_preserves_original_message():
    """The original user message must survive the graph invocation."""
    with patch("simple_ai_bot.llm", _mock_llm()):
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="Test message")]})
        assert result["message"][0].content == "Test message"


def test_invoke_single_message():
    """Graph should handle a single message without errors."""
    with patch("simple_ai_bot.llm", _mock_llm(("ok",))):
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="Hello")]})
        assert result is not None


def test_invoke_empty_token_stream():
    """LLM returning empty tokens should not crash the node."""
    with patch("simple_ai_bot.llm", _mock_llm(())):  # empty stream
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="Hello")]})
        assert "message" in result


# ── State structure ────────────────────────────────────────────────────────

def test_state_key_is_message_not_messages():
    """simple_ai_bot uses 'message' (singular), not 'messages'."""
    with patch("simple_ai_bot.llm", _mock_llm()):
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="hi")]})
        assert "message" in result
        assert "messages" not in result


def test_state_message_is_list():
    """State['message'] should always be a list."""
    with patch("simple_ai_bot.llm", _mock_llm()):
        app = graph.compile()
        result = app.invoke({"message": [HumanMessage(content="hi")]})
        assert isinstance(result["message"], list)


def test_multiple_messages_in_state():
    """State can hold multiple messages — all should be preserved."""
    with patch("simple_ai_bot.llm", _mock_llm(("response",))):
        app = graph.compile()
        msgs = [HumanMessage(content="First"), HumanMessage(content="Second")]
        result = app.invoke({"message": msgs})
        assert len(result["message"]) == 2
