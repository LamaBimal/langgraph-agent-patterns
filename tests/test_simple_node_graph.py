"""
Tests for simple_node_graph.py

Verifies basic graph wiring, state transformation, and edge cases
without requiring Ollama to be running.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simple_node_graph import graph, AgentState


def _compile():
    return graph.compile()


def test_hello_prepended():
    """Node should prepend 'Hello ' to the input message."""
    app = _compile()
    result = app.invoke({"message": "World"})
    assert result["message"] == "Hello World"


def test_hello_with_name():
    """Works with a real name."""
    app = _compile()
    result = app.invoke({"message": "Bimal"})
    assert result["message"] == "Hello Bimal"


def test_empty_string():
    """Empty string input still prepends 'Hello '."""
    app = _compile()
    result = app.invoke({"message": ""})
    assert result["message"] == "Hello "


def test_state_key_preserved():
    """Result dict should contain the 'message' key."""
    app = _compile()
    result = app.invoke({"message": "Test"})
    assert "message" in result


def test_numeric_string():
    """Numeric string input is handled correctly."""
    app = _compile()
    result = app.invoke({"message": "42"})
    assert result["message"] == "Hello 42"
