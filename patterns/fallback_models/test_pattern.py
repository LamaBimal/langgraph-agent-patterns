"""Tests for fallback_models pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import MagicMock, patch
from fallback_pattern import (
    FallbackChain, QualityGate, not_empty, not_refusal,
    min_word_count, ModelAttempt
)
from langchain_core.messages import HumanMessage, AIMessage


def _mock_model(response: str = "Hello", raises=None):
    m = MagicMock()
    m.model = "mock-model"
    if raises:
        m.invoke.side_effect = raises
    else:
        m.invoke.return_value = AIMessage(content=response)
    return m


# ── QualityGate tests ──────────────────────────────────────────────────────

def test_quality_gate_passes_good_response():
    gate = QualityGate(min_length=5)
    assert gate.passes("This is a good response") is True


def test_quality_gate_fails_empty():
    gate = QualityGate()
    assert gate.passes("") is False
    assert gate.passes("   ") is False


def test_quality_gate_fails_too_short():
    gate = QualityGate(min_length=20)
    assert gate.passes("Short") is False


def test_not_empty_check():
    assert not_empty("Hello") is True
    assert not_empty("") is False


def test_not_refusal_check():
    assert not_refusal("Here is the answer...") is True
    assert not_refusal("I cannot help with that") is False
    assert not_refusal("As an AI, I don't...") is False


def test_min_word_count_check():
    check = min_word_count(5)
    assert check("one two three four five") is True
    assert check("one two") is False


# ── FallbackChain tests ────────────────────────────────────────────────────

def test_uses_primary_when_successful():
    primary = _mock_model("Great answer from primary")
    secondary = _mock_model("Secondary answer")
    chain = FallbackChain(models=[primary, secondary])
    response, model = chain.invoke([HumanMessage(content="test")])
    assert response == "Great answer from primary"
    primary.invoke.assert_called_once()
    secondary.invoke.assert_not_called()


def test_falls_back_to_secondary_on_exception():
    primary = _mock_model(raises=ConnectionError("timeout"))
    secondary = _mock_model("Secondary works fine")
    chain = FallbackChain(models=[primary, secondary])
    response, model = chain.invoke([HumanMessage(content="test")])
    assert response == "Secondary works fine"
    assert model == "mock-model"


def test_falls_back_to_static_when_all_fail():
    primary = _mock_model(raises=Exception("fail"))
    secondary = _mock_model(raises=Exception("also fail"))
    chain = FallbackChain(
        models=[primary, secondary],
        static_fallback="Service down",
    )
    response, model = chain.invoke([HumanMessage(content="test")])
    assert response == "Service down"
    assert model == "static_fallback"


def test_falls_back_when_quality_gate_fails():
    primary = _mock_model("No")  # too short
    secondary = _mock_model("This is a proper long answer that passes")
    gate = QualityGate(min_length=20)
    chain = FallbackChain(models=[primary, secondary], quality_gate=gate)
    response, model = chain.invoke([HumanMessage(content="test")])
    assert "proper long answer" in response


def test_summary_contains_model_names():
    primary = _mock_model(raises=Exception("fail"))
    chain = FallbackChain(models=[primary], static_fallback="fallback")
    chain.invoke([HumanMessage(content="test")])
    summary = chain.summary()
    assert "mock-model" in summary
