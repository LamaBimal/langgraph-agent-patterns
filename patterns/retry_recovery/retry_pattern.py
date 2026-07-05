"""
pattern.py — Retry and Error Recovery reusable building block.

Provides:
  - RetryConfig: dataclass for retry policy settings
  - RetryState: TypedDict mixin with retry tracking fields
  - calculate_backoff(): exponential backoff with jitter
  - with_retry(): decorator that wraps any node function with retry logic
  - should_retry(): router function for conditional edges

Usage:
    from patterns.retry_recovery.pattern import (
        RetryConfig, with_retry, should_retry, RetryState
    )
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import TypedDict, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """Policy for retrying a node."""
    max_attempts: int = 3           # total attempts (including first)
    base_delay: float = 1.0         # seconds before first retry
    max_delay: float = 30.0         # cap on wait time
    backoff_multiplier: float = 2.0 # each retry waits multiplier * previous
    jitter: bool = True             # randomise wait to prevent thundering herd
    retryable_exceptions: tuple = (Exception,)  # which exceptions trigger retry


# ── State mixin ────────────────────────────────────────────────────────────

class RetryState(TypedDict, total=False):
    """
    Mixin fields — add these to your AgentState TypedDict.

    Example:
        class MyState(RetryState):
            messages: list
            ...
    """
    attempt: int           # current attempt number (0-indexed)
    last_error: str        # last exception message
    error_history: list    # all errors across attempts


# ── Backoff calculation ────────────────────────────────────────────────────

def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate wait time for a given attempt using exponential backoff.

    attempt=0 → base_delay * multiplier^0 = base_delay
    attempt=1 → base_delay * multiplier^1
    ...capped at max_delay.
    """
    delay = min(
        config.base_delay * (config.backoff_multiplier ** attempt),
        config.max_delay
    )
    if config.jitter:
        # Uniform jitter: randomise between 50% and 100% of calculated delay
        delay = delay * (0.5 + 0.5 * random.random())
    return delay


# ── Router ─────────────────────────────────────────────────────────────────

def should_retry(state: dict, config: RetryConfig) -> str:
    """
    Conditional edge router: returns "retry" or "error_handler".
    Use this as the function in graph.add_conditional_edges().

    Example:
        graph.add_conditional_edges(
            "my_node",
            lambda s: should_retry(s, retry_config),
            {"retry": "my_node", "error_handler": "handle_error"}
        )
    """
    attempt = state.get("attempt", 0)
    if attempt < config.max_attempts - 1:
        return "retry"
    return "error_handler"


# ── Node wrapper ───────────────────────────────────────────────────────────

def with_retry(config: RetryConfig = None):
    """
    Decorator that wraps a LangGraph node function with retry logic.
    On exception: increments attempt, records error, waits, then raises
    so the graph can route via should_retry().

    Usage:
        retry_cfg = RetryConfig(max_attempts=3, base_delay=1.0)

        @with_retry(retry_cfg)
        def my_node(state):
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: dict) -> dict:
            attempt = state.get("attempt", 0)
            error_history = list(state.get("error_history", []))

            try:
                result = fn(state)
                # Success — reset retry counter
                return {**result, "attempt": 0, "last_error": "", "error_history": error_history}

            except config.retryable_exceptions as e:
                error_msg = f"Attempt {attempt + 1}/{config.max_attempts}: {type(e).__name__}: {e}"
                logger.warning(error_msg)
                error_history.append(error_msg)

                # Wait before signalling for retry
                wait = calculate_backoff(attempt, config)
                logger.info(f"Waiting {wait:.2f}s before retry...")
                time.sleep(wait)

                return {
                    "attempt": attempt + 1,
                    "last_error": error_msg,
                    "error_history": error_history,
                }

        return wrapper
    return decorator


# ── Dead-letter / error handler ────────────────────────────────────────────

def build_error_handler(fallback_value: Any = None, notify: Callable = None):
    """
    Build an error handler node that:
    - Logs the full error history
    - Optionally calls a notify function (e.g. send to Slack)
    - Returns a safe fallback value

    Args:
        fallback_value: value to put in state when all retries exhausted
        notify: optional callable(error_history: list) for alerting
    """
    def error_handler(state: dict) -> dict:
        history = state.get("error_history", [])
        last = state.get("last_error", "unknown error")

        logger.error(f"All retries exhausted. Last error: {last}")
        for i, err in enumerate(history):
            logger.error(f"  [{i+1}] {err}")

        if notify:
            try:
                notify(history)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

        return {
            "last_error": last,
            "error_history": history,
            "result": fallback_value,
        }

    return error_handler
