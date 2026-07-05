"""
pattern.py — Fallback Models reusable building block.

Provides:
  - FallbackChain: tries models in order, returns first successful response
  - QualityGate: checks response quality before accepting it
  - ModelAttempt: dataclass recording each attempt's outcome

Usage:
    from patterns.fallback_models.pattern import FallbackChain, QualityGate

    chain = FallbackChain(models=[primary_llm, secondary_llm, fallback_llm])
    response, used_model = chain.invoke(messages)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)


# ── Attempt record ─────────────────────────────────────────────────────────

@dataclass
class ModelAttempt:
    model_name: str
    success: bool
    response: str = ""
    error: str = ""
    latency_ms: float = 0.0


# ── Quality gate ───────────────────────────────────────────────────────────

class QualityGate:
    """
    Checks whether a model response meets quality criteria.
    Compose multiple checks by passing a list of check functions.

    Each check function: (response: str) → bool  (True = passes)
    """

    def __init__(self, checks: list[Callable[[str], bool]] = None,
                 min_length: int = 10):
        self.checks = checks or []
        self.min_length = min_length

    def passes(self, response: str) -> bool:
        if not response or len(response.strip()) < self.min_length:
            return False
        for check in self.checks:
            if not check(response):
                return False
        return True


# Built-in quality checks
def not_empty(response: str) -> bool:
    return bool(response and response.strip())

def not_refusal(response: str) -> bool:
    """Detect common LLM refusal phrases."""
    refusals = [
        "i cannot", "i can't", "i'm not able to",
        "as an ai", "i don't have the ability",
    ]
    lower = response.lower()
    return not any(r in lower for r in refusals)

def min_word_count(n: int) -> Callable[[str], bool]:
    return lambda r: len(r.split()) >= n


# ── Fallback chain ─────────────────────────────────────────────────────────

class FallbackChain:
    """
    Tries models in order. Returns the first response that passes the
    quality gate. Falls back to a static message if all models fail.

    Args:
        models: list of LangChain chat model instances (in priority order)
        quality_gate: QualityGate instance (default: non-empty response)
        static_fallback: response to return if all models fail
        retry_on_exceptions: tuple of exception types that trigger fallback
    """

    def __init__(
        self,
        models: list,
        quality_gate: QualityGate = None,
        static_fallback: str = "I'm sorry, I'm unable to respond right now.",
        retry_on_exceptions: tuple = (Exception,),
    ):
        self.models = models
        self.quality_gate = quality_gate or QualityGate()
        self.static_fallback = static_fallback
        self.retry_on_exceptions = retry_on_exceptions
        self.attempts: list[ModelAttempt] = []

    def invoke(self, messages: list[BaseMessage]) -> tuple[str, str]:
        """
        Try each model in sequence.

        Returns:
            (response_text, model_name_used)
        """
        self.attempts = []

        for model in self.models:
            model_name = getattr(model, 'model', str(model))
            start = time.monotonic()

            try:
                logger.info(f"Trying model: {model_name}")
                response = model.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                latency = (time.monotonic() - start) * 1000

                if self.quality_gate.passes(content):
                    self.attempts.append(ModelAttempt(
                        model_name=model_name,
                        success=True,
                        response=content,
                        latency_ms=latency,
                    ))
                    logger.info(f"  ✓ {model_name} succeeded ({latency:.0f}ms)")
                    return content, model_name
                else:
                    self.attempts.append(ModelAttempt(
                        model_name=model_name,
                        success=False,
                        response=content,
                        error="Failed quality gate",
                        latency_ms=latency,
                    ))
                    logger.warning(f"  ✗ {model_name} failed quality gate")

            except self.retry_on_exceptions as e:
                latency = (time.monotonic() - start) * 1000
                self.attempts.append(ModelAttempt(
                    model_name=model_name,
                    success=False,
                    error=str(e),
                    latency_ms=latency,
                ))
                logger.warning(f"  ✗ {model_name} raised {type(e).__name__}: {e}")

        # All models failed
        logger.error("All models in fallback chain failed. Using static fallback.")
        return self.static_fallback, "static_fallback"

    def summary(self) -> str:
        """Return a human-readable summary of all attempts."""
        lines = ["Fallback chain summary:"]
        for a in self.attempts:
            status = "✓" if a.success else "✗"
            lines.append(f"  {status} {a.model_name}: {a.error or 'OK'} ({a.latency_ms:.0f}ms)")
        return "\n".join(lines)


# ── LangGraph node builder ─────────────────────────────────────────────────

def build_fallback_node(chain: FallbackChain, state_key: str = "messages"):
    """
    Build a LangGraph node that uses a FallbackChain.

    Args:
        chain: FallbackChain instance
        state_key: key in state that holds the message list

    Returns:
        node function compatible with StateGraph.add_node()
    """
    def node(state: dict) -> dict:
        messages = state.get(state_key, [])
        response, model_used = chain.invoke(messages)
        logger.info(f"Response from: {model_used}")
        print(f"\n  [fallback] Used model: {model_used}")
        print(chain.summary())
        return {state_key: [AIMessage(content=response)], "model_used": model_used}

    return node
