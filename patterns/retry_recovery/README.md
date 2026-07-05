# Pattern: Retry and Error Recovery

## What It Is
Retry and error recovery makes agents resilient to transient failures —
network timeouts, LLM rate limits, tool errors — by automatically retrying
with exponential backoff, tracking attempt counts in state, and routing
to a recovery path when retries are exhausted.

## When to Use
- Tool calls that hit external APIs (search, databases, webhooks)
- LLM calls that may time out or hit rate limits
- Any step where transient failures are expected but eventual success is likely
- Workflows where a fallback behaviour (cached result, default value) is
  acceptable when all retries fail

## How It Works

```
Node fails
    ↓
Increment attempt counter in state
    ↓
attempts < max_retries?
    ├── Yes → wait (exponential backoff) → retry node
    └── No  → route to error_handler node
                  ↓
            return safe default / notify / log
```

## Key Concepts
- Attempt counter stored in `AgentState` — survives graph loops
- Exponential backoff: `min(base * 2^attempt, max_wait)` seconds
- Jitter added to prevent thundering herd on shared resources
- Dead-letter state: captures the last error for observability

## Files
- `pattern.py` — `RetryConfig`, `with_retry` node wrapper, backoff utilities
- `example.py` — web search agent with retry on tool failure
- `test_pattern.py` — tests for backoff calculation, routing, exhaustion

## Tradeoffs
| Pro | Con |
|-----|-----|
| Handles transient failures automatically | Adds latency on retries |
| Configurable per-node retry policy | Masking bugs if retried too aggressively |
| Dead-letter captures errors for debugging | State grows with each attempt |
