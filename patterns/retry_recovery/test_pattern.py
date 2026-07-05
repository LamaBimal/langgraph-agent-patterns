"""Tests for retry_recovery pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from retry_pattern import RetryConfig, calculate_backoff, should_retry, build_error_handler


def test_backoff_increases_with_attempt():
    cfg = RetryConfig(base_delay=1.0, backoff_multiplier=2.0, jitter=False)
    assert calculate_backoff(0, cfg) == 1.0
    assert calculate_backoff(1, cfg) == 2.0
    assert calculate_backoff(2, cfg) == 4.0


def test_backoff_capped_at_max_delay():
    cfg = RetryConfig(base_delay=1.0, max_delay=5.0, backoff_multiplier=2.0, jitter=False)
    assert calculate_backoff(10, cfg) == 5.0


def test_backoff_with_jitter_within_range():
    cfg = RetryConfig(base_delay=2.0, backoff_multiplier=2.0, jitter=True)
    for _ in range(20):
        val = calculate_backoff(0, cfg)
        assert 1.0 <= val <= 2.0  # 50%-100% of 2.0


def test_should_retry_when_attempts_remain():
    cfg = RetryConfig(max_attempts=3)
    state = {"attempt": 0}
    assert should_retry(state, cfg) == "retry"
    state = {"attempt": 1}
    assert should_retry(state, cfg) == "retry"


def test_should_route_to_error_handler_when_exhausted():
    cfg = RetryConfig(max_attempts=3)
    state = {"attempt": 2}
    assert should_retry(state, cfg) == "error_handler"


def test_error_handler_returns_fallback():
    handler = build_error_handler(fallback_value="Service unavailable")
    result = handler({"error_history": ["err1"], "last_error": "err1"})
    assert result["result"] == "Service unavailable"


def test_error_handler_calls_notify():
    notified = []
    handler = build_error_handler(notify=lambda h: notified.append(h))
    handler({"error_history": ["e1", "e2"], "last_error": "e2"})
    assert len(notified) == 1
    assert len(notified[0]) == 2


def test_retry_config_defaults():
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.base_delay == 1.0
    assert cfg.jitter is True
