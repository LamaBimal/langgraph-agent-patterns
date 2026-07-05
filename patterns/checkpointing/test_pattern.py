"""Tests for checkpointing pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from checkpoint_pattern import CheckpointerFactory, thread_config
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class SimpleState(TypedDict):
    value: int


def increment(state: SimpleState) -> SimpleState:
    return {"value": state["value"] + 1}


def _build_app(checkpointer):
    g = StateGraph(SimpleState)
    g.add_node("inc", increment)
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    return g.compile(checkpointer=checkpointer)


def test_thread_config_structure():
    cfg = thread_config("user-123")
    assert cfg["configurable"]["thread_id"] == "user-123"


def test_thread_config_with_extras():
    cfg = thread_config("abc", checkpoint_id="xyz")
    assert cfg["configurable"]["checkpoint_id"] == "xyz"


def test_memory_checkpointer_type():
    cp = CheckpointerFactory.memory()
    assert isinstance(cp, MemorySaver)


def test_memory_persists_within_process():
    cp = CheckpointerFactory.memory()
    app = _build_app(cp)
    cfg = thread_config("test-thread-1")

    result1 = app.invoke({"value": 0}, config=cfg)
    assert result1["value"] == 1

    # Same thread — state is remembered
    result2 = app.invoke({"value": result1["value"]}, config=cfg)
    assert result2["value"] == 2


def test_different_threads_isolated():
    cp = CheckpointerFactory.memory()
    app = _build_app(cp)

    r1 = app.invoke({"value": 10}, config=thread_config("thread-A"))
    r2 = app.invoke({"value": 20}, config=thread_config("thread-B"))

    assert r1["value"] == 11
    assert r2["value"] == 21


def test_auto_uses_memory_in_test_env(monkeypatch):
    monkeypatch.setenv("TESTING", "true")
    cp = CheckpointerFactory.auto()
    assert isinstance(cp, MemorySaver)
