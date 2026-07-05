"""
Tests for memory_agent.py

Verifies graph structure, state accumulation, message types,
and MemorySaver thread isolation. No Ollama required — LLM is mocked.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from memory_agent import graph, AgentState, process_message


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm_stream(tokens=("Hello", " world")):
    """Return a mock LLM whose .stream() yields token chunks."""
    chunks = [MagicMock(content=t) for t in tokens]
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter(chunks)
    return mock_llm


def _compile_fresh():
    return graph.compile(checkpointer=MemorySaver())


# ── Graph structure ─────────────────────────────────────────────────────────

def test_graph_has_process_node():
    node_names = list(graph.compile().get_graph().nodes.keys())
    assert "process" in node_names


def test_graph_has_start_and_end():
    nodes = list(graph.compile().get_graph().nodes.keys())
    assert "__start__" in nodes
    assert "__end__" in nodes


def test_graph_compiles_without_error():
    assert graph.compile() is not None


def test_graph_compiles_with_checkpointer():
    assert _compile_fresh() is not None


# ── process_message node ───────────────────────────────────────────────────

def test_process_message_appends_ai_message():
    """process_message should append exactly one AIMessage to state."""
    with patch("memory_agent.llm", _mock_llm_stream(("Hi",))):
        state: AgentState = {"messages": [HumanMessage(content="Hello")]}
        result = process_message(state)
        assert isinstance(result["messages"][-1], AIMessage)


def test_process_message_content_matches_tokens():
    """AIMessage content should be the joined token stream."""
    with patch("memory_agent.llm", _mock_llm_stream(("Hello", " world"))):
        state: AgentState = {"messages": [HumanMessage(content="Hi")]}
        result = process_message(state)
        assert result["messages"][-1].content == "Hello world"


def test_process_message_preserves_history():
    """Existing messages should remain after process_message runs."""
    with patch("memory_agent.llm", _mock_llm_stream(("OK",))):
        original = [HumanMessage(content="Q1"), AIMessage(content="A1")]
        state: AgentState = {"messages": list(original)}
        result = process_message(state)
        assert result["messages"][0].content == "Q1"
        assert result["messages"][1].content == "A1"


# ── Full graph invocation ──────────────────────────────────────────────────

def test_invoke_returns_messages_key():
    with patch("memory_agent.llm", _mock_llm_stream(("Hi",))):
        app = graph.compile()
        result = app.invoke({"messages": [HumanMessage(content="test")]})
        assert "messages" in result


def test_invoke_produces_ai_message():
    with patch("memory_agent.llm", _mock_llm_stream(("response",))):
        app = graph.compile()
        result = app.invoke({"messages": [HumanMessage(content="test")]})
        assert any(isinstance(m, AIMessage) for m in result["messages"])


def test_invoke_message_count():
    """One human + one AI = 2 messages total after a single turn."""
    with patch("memory_agent.llm", _mock_llm_stream(("reply",))):
        app = graph.compile()
        result = app.invoke({"messages": [HumanMessage(content="hello")]})
        assert len(result["messages"]) == 2


# ── Multi-turn and thread isolation ───────────────────────────────────────

def test_multi_turn_accumulates_history():
    """Second turn state built from first result should have 4 messages."""
    with patch("memory_agent.llm", _mock_llm_stream(("ok",))):
        app = _compile_fresh()
        cfg = {"configurable": {"thread_id": "mt-test"}}

        r1 = app.invoke({"messages": [HumanMessage(content="Turn 1")]}, config=cfg)
        assert len(r1["messages"]) == 2

        r2 = app.invoke(
            {"messages": r1["messages"] + [HumanMessage(content="Turn 2")]},
            config=cfg,
        )
        assert len(r2["messages"]) == 4


def test_different_threads_are_isolated():
    """Two different thread_ids must not share state."""
    with patch("memory_agent.llm", _mock_llm_stream(("A",))):
        app = _compile_fresh()

        r_a = app.invoke(
            {"messages": [HumanMessage(content="Thread A")]},
            config={"configurable": {"thread_id": "thread-A"}},
        )
        r_b = app.invoke(
            {"messages": [HumanMessage(content="Thread B")]},
            config={"configurable": {"thread_id": "thread-B"}},
        )

        # Both threads should only have their own 2 messages
        assert len(r_a["messages"]) == 2
        assert len(r_b["messages"]) == 2
        assert r_a["messages"][0].content == "Thread A"
        assert r_b["messages"][0].content == "Thread B"
