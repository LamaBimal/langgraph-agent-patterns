"""
pattern.py — Streaming reusable building block.

Provides helpers for each LangGraph streaming mode plus an
SSE (Server-Sent Events) formatter for web delivery.

Streaming modes:
  - stream_tokens():  token-by-token LLM output ("messages" mode)
  - stream_steps():   full state after each node ("values" mode)
  - stream_updates(): delta changes per node ("updates" mode)
  - stream_events():  all graph events ("events" mode)
  - to_sse():         format any stream as SSE for HTTP/WebSocket

Usage:
    from patterns.streaming.pattern import stream_tokens, stream_steps

    for token in stream_tokens(app, state, config):
        print(token, end="", flush=True)
"""

import json
from typing import Any, Generator, Iterator
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk


# ── Token streaming ────────────────────────────────────────────────────────

def stream_tokens(
    app,
    initial_state: dict,
    config: dict = None,
    node_name: str = None,
) -> Generator[str, None, None]:
    """
    Stream tokens from LLM nodes in the graph.

    Uses LangGraph's "messages" streaming mode which yields
    (message_chunk, metadata) tuples from LLM nodes.

    Args:
        app: compiled LangGraph graph
        initial_state: initial state dict
        config: optional LangGraph config (thread_id, etc.)
        node_name: if set, only yield tokens from this node

    Yields:
        str: individual token strings
    """
    kwargs = {"stream_mode": "messages"}
    if config:
        kwargs["config"] = config

    for chunk, metadata in app.stream(initial_state, **kwargs):
        # Filter by node name if specified
        if node_name and metadata.get("langgraph_node") != node_name:
            continue
        if isinstance(chunk, (AIMessage, AIMessageChunk)):
            if chunk.content:
                yield chunk.content


# ── Step streaming ─────────────────────────────────────────────────────────

def stream_steps(
    app,
    initial_state: dict,
    config: dict = None,
) -> Generator[dict, None, None]:
    """
    Yield the full graph state after each node completes.

    Useful for showing step-by-step progress in a UI or CLI.

    Yields:
        dict: {"node": node_name, "state": full_state_values}
    """
    kwargs = {"stream_mode": "values"}
    if config:
        kwargs["config"] = config

    prev_keys: set = set()
    for state in app.stream(initial_state, **kwargs):
        # Detect which key changed to infer the node that just ran
        new_keys = set(state.keys()) - prev_keys
        prev_keys = set(state.keys())
        yield {"state": state}


# ── Update streaming ───────────────────────────────────────────────────────

def stream_updates(
    app,
    initial_state: dict,
    config: dict = None,
) -> Generator[dict, None, None]:
    """
    Yield only the delta (changed keys) after each node, not the full state.

    More efficient than stream_steps for large state objects.

    Yields:
        dict: {"node": node_name, "updates": delta_dict}
    """
    kwargs = {"stream_mode": "updates"}
    if config:
        kwargs["config"] = config

    for update in app.stream(initial_state, **kwargs):
        for node_name, delta in update.items():
            yield {"node": node_name, "updates": delta}


# ── Event streaming ────────────────────────────────────────────────────────

def stream_events(
    app,
    initial_state: dict,
    config: dict = None,
    include_types: list[str] = None,
) -> Generator[dict, None, None]:
    """
    Yield all graph events (on_chain_start, on_tool_start, on_llm_stream, etc.)

    Args:
        include_types: filter to specific event types (e.g. ["on_llm_stream"])

    Yields:
        dict: LangChain event dict with "event", "name", "data" keys
    """
    kwargs = {}
    if config:
        kwargs["config"] = config

    for event in app.astream_events(initial_state, version="v2", **kwargs):
        if include_types and event.get("event") not in include_types:
            continue
        yield event


# ── Console streaming helpers ──────────────────────────────────────────────

def print_tokens(app, initial_state: dict, config: dict = None, prefix: str = "") -> str:
    """
    Stream tokens to stdout in real-time. Returns the full accumulated response.

    Args:
        prefix: string to print before streaming starts (e.g. "Assistant: ")
    """
    if prefix:
        print(prefix, end="", flush=True)

    full_response = ""
    for token in stream_tokens(app, initial_state, config):
        print(token, end="", flush=True)
        full_response += token
    print()  # newline after stream
    return full_response


def print_steps(app, initial_state: dict, config: dict = None) -> dict:
    """
    Print a progress indicator for each completed node. Returns final state.
    """
    final_state = {}
    for step in stream_updates(app, initial_state, config):
        node = step["node"]
        updates = step["updates"]
        changed_keys = list(updates.keys()) if isinstance(updates, dict) else []
        print(f"  ✓ {node} → updated: {changed_keys}")
        final_state.update(updates if isinstance(updates, dict) else {})
    return final_state


# ── SSE formatter for web delivery ─────────────────────────────────────────

def to_sse(data: Any, event_type: str = "message") -> str:
    """
    Format data as a Server-Sent Event string for HTTP streaming.

    Compatible with EventSource API in browsers.

    Args:
        data: any JSON-serialisable value
        event_type: SSE event type field

    Returns:
        str: formatted SSE string
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data)

    return f"event: {event_type}\ndata: {payload}\n\n"


def sse_token_stream(
    app,
    initial_state: dict,
    config: dict = None,
) -> Generator[str, None, None]:
    """
    Yield SSE-formatted token strings suitable for an HTTP streaming response.

    Usage (FastAPI):
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            sse_token_stream(app, state),
            media_type="text/event-stream"
        )
    """
    for token in stream_tokens(app, initial_state, config):
        yield to_sse({"token": token}, event_type="token")
    yield to_sse({"done": True}, event_type="done")
