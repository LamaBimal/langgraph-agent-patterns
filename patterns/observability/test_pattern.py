"""Tests for observability pattern."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from observability_pattern import AgentTracer, TokenCounter, TokenUsage, NodeEvent, traced_node


def test_tracer_records_events():
    tracer = AgentTracer("test-session")
    tracer.record(NodeEvent("node1", "success", latency_ms=50.0))
    assert len(tracer.events) == 1
    assert tracer.events[0].node_name == "node1"


def test_tracer_records_error():
    tracer = AgentTracer("test-session")
    tracer.record(NodeEvent("node1", "error", error="timeout"))
    assert tracer.events[0].status == "error"
    assert tracer.events[0].error == "timeout"


def test_token_counter_totals():
    tc = TokenCounter()
    tc.record("node1", "a" * 400, 100.0)   # ≈ 100 tokens
    tc.record("node2", "b" * 800, 200.0)   # ≈ 200 tokens
    assert tc.total_tokens == 300
    assert tc.total_latency_ms == 300.0


def test_token_counter_by_node():
    tc = TokenCounter()
    tc.record("llm", "x" * 400, 50.0)
    tc.record("tool", "y" * 200, 10.0)
    by_node = tc.by_node()
    assert "llm" in by_node
    assert "tool" in by_node


def test_traced_node_records_success():
    tracer = AgentTracer("test")

    @traced_node(tracer, "test_node")
    def my_node(state):
        return {"result": "ok"}

    my_node({"input": "test"})

    events = [e for e in tracer.events if e.status in ("success", "start")]
    assert any(e.status == "success" for e in events)


def test_traced_node_records_error():
    tracer = AgentTracer("test")

    @traced_node(tracer, "error_node")
    def bad_node(state):
        raise ValueError("test error")

    try:
        bad_node({})
    except ValueError:
        pass

    assert any(e.status == "error" for e in tracer.events)
    assert any("test error" in e.error for e in tracer.events)


def test_traced_node_measures_latency():
    tracer = AgentTracer("test")

    @traced_node(tracer, "slow_node")
    def slow_node(state):
        time.sleep(0.05)
        return {}

    slow_node({})
    success_events = [e for e in tracer.events if e.status == "success"]
    assert success_events[0].latency_ms >= 40  # at least 40ms
