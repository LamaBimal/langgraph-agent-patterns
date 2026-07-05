"""Tests for streaming pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import json
from streaming_pattern import to_sse, stream_updates, stream_steps
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ── SSE formatter ──────────────────────────────────────────────────────────

def test_sse_string_data():
    result = to_sse("hello", event_type="message")
    assert "event: message" in result
    assert "data: hello" in result
    assert result.endswith("\n\n")


def test_sse_dict_data():
    result = to_sse({"token": "hi"}, event_type="token")
    assert "event: token" in result
    data_line = [l for l in result.split("\n") if l.startswith("data:")][0]
    payload = json.loads(data_line.replace("data: ", ""))
    assert payload["token"] == "hi"


def test_sse_done_event():
    result = to_sse({"done": True}, event_type="done")
    assert "event: done" in result


def test_sse_format_has_double_newline():
    result = to_sse("test")
    assert result.endswith("\n\n")


# ── Stream updates ─────────────────────────────────────────────────────────

class Counter(TypedDict):
    count: int
    label: str


def _build_counter_app():
    def step1(state: Counter) -> Counter:
        return {"count": state["count"] + 1, "label": "step1"}

    def step2(state: Counter) -> Counter:
        return {"count": state["count"] + 10, "label": "step2"}

    g = StateGraph(Counter)
    g.add_node("step1", step1)
    g.add_node("step2", step2)
    g.add_edge(START, "step1")
    g.add_edge("step1", "step2")
    g.add_edge("step2", END)
    return g.compile()


def test_stream_updates_yields_node_names():
    app = _build_counter_app()
    updates = list(stream_updates(app, {"count": 0, "label": ""}))
    node_names = [u["node"] for u in updates]
    assert "step1" in node_names
    assert "step2" in node_names


def test_stream_updates_contains_deltas():
    app = _build_counter_app()
    updates = list(stream_updates(app, {"count": 0, "label": ""}))
    # step1 should update count to 1
    step1_update = next(u for u in updates if u["node"] == "step1")
    assert step1_update["updates"]["count"] == 1


def test_stream_steps_yields_states():
    app = _build_counter_app()
    steps = list(stream_steps(app, {"count": 0, "label": ""}))
    assert len(steps) > 0
    assert "state" in steps[-1]
