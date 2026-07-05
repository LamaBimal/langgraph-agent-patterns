# Pattern: Streaming

## What It Is
Streaming delivers agent output incrementally — tokens, node events, or
tool results — as they are produced rather than waiting for the full
response. This dramatically improves perceived responsiveness.

## Streaming Modes (LangGraph)

| Mode | What you receive | Use case |
|------|-----------------|----------|
| `"values"` | Full state after each node | Step-by-step progress |
| `"updates"` | Delta changes per node | Efficient state monitoring |
| `"messages"` | Token-by-token from LLM nodes | Chat-style streaming |
| `"events"` | All graph events (start, end, tool) | Full observability |
| `"debug"` | Verbose internal events | Debugging |

## When to Use
- Chat interfaces (users see text appear in real time)
- Long-running agents (show progress so users know it's working)
- Tool-heavy agents (show which tools are being called)
- SSE / WebSocket backends serving a frontend

## Files
- `pattern.py` — streaming helpers for each mode, SSE formatter
- `example.py` — streaming chat agent with live token output
- `test_pattern.py` — tests for stream output collection and formatting

## Tradeoffs
| Pro | Con |
|-----|-----|
| Dramatically better UX for long responses | Slightly more complex client code |
| Users see progress on multi-step agents | Tool results may disrupt token stream |
| Works with existing graph structure | SSE requires HTTP server for web delivery |
