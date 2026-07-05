"""Tests for multi_model pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import MagicMock, patch
from multimodel_pattern import SpecialistAgent, ModelRouter, IntentClassifier, build_orchestrator
from langchain_core.messages import AIMessage, HumanMessage


def _mock_llm(response_text: str):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=response_text)
    return llm


# ── IntentClassifier ───────────────────────────────────────────────────────

def test_classifier_valid_intent():
    llm = _mock_llm('{"intent": "math", "confidence": 0.9, "reasoning": "math problem"}')
    clf = IntentClassifier(llm=llm, categories=["math", "general"])
    intent, conf = clf.classify("What is 2 + 2?")
    assert intent == "math"
    assert conf == 0.9


def test_classifier_unknown_intent_fallback():
    llm = _mock_llm('{"intent": "unknown_category", "confidence": 0.5, "reasoning": "..."}')
    clf = IntentClassifier(llm=llm, categories=["math", "general"], default="general")
    intent, _ = clf.classify("some query")
    assert intent == "general"


def test_classifier_fallback_on_json_error():
    llm = _mock_llm("not valid json at all")
    clf = IntentClassifier(llm=llm, categories=["math", "general"], default="general")
    intent, conf = clf.classify("some query")
    assert intent == "general"
    assert conf == 0.0


# ── SpecialistAgent ────────────────────────────────────────────────────────

def test_specialist_invokes_llm():
    llm = _mock_llm("42 is the answer")
    spec = SpecialistAgent(name="math", llm=llm, system_prompt="You are a math expert.")
    result = spec.invoke("What is 6 * 7?")
    assert result == "42 is the answer"
    llm.invoke.assert_called_once()


def test_specialist_includes_context():
    llm = _mock_llm("response")
    spec = SpecialistAgent(name="general", llm=llm, system_prompt="Be helpful.")
    spec.invoke("query", context=[HumanMessage(content="prior message")])
    call_args = llm.invoke.call_args[0][0]
    # Should have system + context + new query
    assert len(call_args) >= 3


# ── ModelRouter ────────────────────────────────────────────────────────────

def test_router_dispatches_to_correct_specialist():
    classifier_llm = _mock_llm('{"intent": "math", "confidence": 0.9, "reasoning": ""}')
    math_llm = _mock_llm("The answer is 4")
    math_spec = SpecialistAgent("math", math_llm, "You are a math expert.")

    general_llm = _mock_llm("General response")
    general_spec = SpecialistAgent("general", general_llm, "Be helpful.")

    router = ModelRouter(
        classifier_llm=classifier_llm,
        routes={"math": math_spec, "general": general_spec},
        default_route="general",
    )
    response, specialist = router.route("What is 2 + 2?")
    assert specialist == "math"
    assert "4" in response


def test_router_uses_default_on_low_confidence():
    classifier_llm = _mock_llm('{"intent": "math", "confidence": 0.1, "reasoning": ""}')
    math_spec = SpecialistAgent("math", _mock_llm("math answer"), "expert")
    general_spec = SpecialistAgent("general", _mock_llm("general answer"), "helpful")

    router = ModelRouter(
        classifier_llm=classifier_llm,
        routes={"math": math_spec, "general": general_spec},
        default_route="general",
        confidence_threshold=0.5,
    )
    response, specialist = router.route("vague query")
    assert specialist == "general"


# ── Orchestrator graph ─────────────────────────────────────────────────────

def test_orchestrator_compiles():
    classifier_llm = _mock_llm('{"intent": "general", "confidence": 0.8, "reasoning": ""}')
    general_spec = SpecialistAgent("general", _mock_llm("ok"), "helpful")
    router = ModelRouter(
        classifier_llm=classifier_llm,
        routes={"general": general_spec},
        default_route="general",
    )
    app = build_orchestrator(router)
    assert app is not None
