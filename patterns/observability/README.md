# Pattern: Observability

## What It Is
Observability gives you visibility into what your agents are doing —
which nodes ran, how long they took, what tokens were consumed, and
where failures occurred. Without it, debugging production agents is guesswork.

## Three Pillars

| Pillar | What | Tools |
|--------|------|-------|
| **Logging** | Structured node-level events | Python logging, JSON logs |
| **Tracing** | Full execution graph with inputs/outputs | LangSmith, OpenTelemetry |
| **Metrics** | Latency, token usage, error rate | Custom counters, Prometheus |

## When to Use
- Always — especially before going to production
- Debugging unexpected agent behaviour
- Cost monitoring (token usage per session)
- SLA tracking (p95 latency per node)

## Files
- `pattern.py` — `AgentTracer`, `NodeTimer`, structured log formatter
- `example.py` — traced agent with per-node timing and token counting
- `test_pattern.py` — tests for tracing, timing, log output

## Tradeoffs
| Pro | Con |
|-----|-----|
| Fast debugging and root cause analysis | LangSmith requires API key |
| Token cost visibility | Adds minor overhead to each node |
| Audit trail for compliance | Log storage cost at scale |
