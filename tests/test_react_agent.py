"""
Tests for ReAct_agent.py — math tools and graph structure.

Math tools are pure functions and can be tested without Ollama.
Graph structure tests verify nodes/edges are wired correctly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ReAct_agent import add, subtract, multiply, agent, AgentState


# ── Math tool tests ────────────────────────────────────────────────────────

def test_add_positive():
    assert add.invoke({"a": 5, "b": 10}) == 15


def test_add_negative():
    assert add.invoke({"a": -3, "b": 7}) == 4


def test_add_zeros():
    assert add.invoke({"a": 0, "b": 0}) == 0


def test_subtract_basic():
    assert subtract.invoke({"a": 20, "b": 8}) == 12


def test_subtract_negative_result():
    assert subtract.invoke({"a": 5, "b": 10}) == -5


def test_multiply_basic():
    assert multiply.invoke({"a": 4, "b": 5}) == 20


def test_multiply_by_zero():
    assert multiply.invoke({"a": 100, "b": 0}) == 0


def test_multiply_negative():
    assert multiply.invoke({"a": -3, "b": 4}) == -12


# ── Graph structure tests ──────────────────────────────────────────────────

def test_agent_has_nodes():
    """Agent graph should contain 'our_agent' and 'tools' nodes."""
    node_names = list(agent.get_graph().nodes.keys())
    assert "our_agent" in node_names
    assert "tools" in node_names


def test_agent_compiled():
    """Agent should compile without error."""
    assert agent is not None
