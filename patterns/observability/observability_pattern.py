"""
pattern.py — Observability reusable building block.

Provides:
  - AgentTracer: wraps node functions with timing and structured logging
  - TokenCounter: tracks estimated token usage per node and session
  - ObservabilityState: TypedDict mixin with trace fields
  - traced_node(): decorator for automatic node instrumentation

Usage:
    from patterns.observability.pattern import AgentTracer, traced_node

    tracer = AgentTracer(session_id="abc")

    @traced_node(tracer, node_name="my_node")
    def my_node(state):
        ...
"""

import time
import logging
import json
from dataclasses import dataclass, field
from typing import Callable, TypedDict
from functools import wraps

# Configure structured JSON logging
logger = logging.getLogger("agent.observability")


def setup_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Configure logging. Call once at app startup."""
    handler = logging.StreamHandler()
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler], force=True)


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON for structured log aggregators."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)


# ── Token counter ──────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    node_name: str
    estimated_tokens: int
    latency_ms: float


class TokenCounter:
    """
    Tracks token usage across nodes in a session.
    Uses rough estimation (4 chars ≈ 1 token) since Ollama doesn't
    always return exact usage metadata.
    """

    def __init__(self):
        self._usage: list[TokenUsage] = []

    def record(self, node_name: str, text: str, latency_ms: float) -> None:
        tokens = max(1, len(text) // 4)
        self._usage.append(TokenUsage(node_name, tokens, latency_ms))

    @property
    def total_tokens(self) -> int:
        return sum(u.estimated_tokens for u in self._usage)

    @property
    def total_latency_ms(self) -> float:
        return sum(u.latency_ms for u in self._usage)

    def by_node(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for u in self._usage:
            result[u.node_name] = result.get(u.node_name, 0) + u.estimated_tokens
        return result

    def summary(self) -> str:
        lines = [f"Token usage summary (estimated):"]
        for node, tokens in self.by_node().items():
            lines.append(f"  {node}: ~{tokens} tokens")
        lines.append(f"  TOTAL: ~{self.total_tokens} tokens | "
                     f"{self.total_latency_ms:.0f}ms total latency")
        return "\n".join(lines)


# ── Observability state mixin ──────────────────────────────────────────────

class ObservabilityState(TypedDict, total=False):
    """Add these fields to your AgentState."""
    trace_events: list      # list of node execution events
    session_id: str
    total_tokens: int


# ── Agent tracer ───────────────────────────────────────────────────────────

@dataclass
class NodeEvent:
    node_name: str
    status: str              # "start" | "success" | "error"
    latency_ms: float = 0.0
    error: str = ""
    metadata: dict = field(default_factory=dict)


class AgentTracer:
    """
    Records execution events for each node in the graph.
    Provides a full timeline of the agent's execution.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.events: list[NodeEvent] = []
        self.token_counter = TokenCounter()

    def record(self, event: NodeEvent) -> None:
        self.events.append(event)
        log_data = {
            "session_id": self.session_id,
            "node": event.node_name,
            "status": event.status,
            "latency_ms": round(event.latency_ms, 2),
        }
        if event.error:
            log_data["error"] = event.error
        if event.metadata:
            log_data.update(event.metadata)

        level = logging.ERROR if event.status == "error" else logging.INFO
        logger.log(level, f"[{event.node_name}] {event.status}", extra={"extra": log_data})

    def print_timeline(self) -> None:
        """Print a human-readable execution timeline."""
        print(f"\n{'─'*55}")
        print(f"Execution timeline | session: {self.session_id}")
        print(f"{'─'*55}")
        for e in self.events:
            icon = "✓" if e.status == "success" else "✗" if e.status == "error" else "→"
            print(f"  {icon} {e.node_name:<20} {e.status:<10} {e.latency_ms:>7.1f}ms")
        print(f"{'─'*55}")
        print(self.token_counter.summary())
        print(f"{'─'*55}\n")


# ── Node decorator ─────────────────────────────────────────────────────────

def traced_node(tracer: AgentTracer, node_name: str = None):
    """
    Decorator: wraps a LangGraph node function with automatic tracing.

    Records start time, end time, status, and any exceptions.
    Token usage is estimated from the response content if present.

    Usage:
        @traced_node(tracer, "my_llm_node")
        def my_node(state):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        name = node_name or fn.__name__

        @wraps(fn)
        def wrapper(state: dict) -> dict:
            start = time.monotonic()
            tracer.record(NodeEvent(node_name=name, status="start"))
            try:
                result = fn(state)
                latency = (time.monotonic() - start) * 1000

                # Estimate tokens from any string values in result
                response_text = ""
                if isinstance(result, dict):
                    for v in result.values():
                        if isinstance(v, list):
                            for item in v:
                                response_text += str(getattr(item, 'content', item))

                tracer.token_counter.record(name, response_text, latency)
                tracer.record(NodeEvent(node_name=name, status="success", latency_ms=latency))
                return result

            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                tracer.record(NodeEvent(
                    node_name=name,
                    status="error",
                    latency_ms=latency,
                    error=str(e),
                ))
                raise

        return wrapper
    return decorator
