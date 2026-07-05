# Pattern: Human-in-the-Loop

## What It Is
Human-in-the-loop (HITL) pauses graph execution at a specific node and waits
for a human to review, approve, modify, or reject the agent's planned action
before it continues.

LangGraph supports this natively via **interrupt_before** / **interrupt_after**
on `graph.compile()`, combined with a persistent checkpointer so state is
preserved across the pause.

## When to Use
- Before executing irreversible actions (sending emails, deleting records)
- When the agent's confidence is low and a human review improves quality
- Regulated workflows requiring an audit trail of human approvals
- Agentic systems writing or modifying production data

## How It Works

```
User Input → LLM drafts action → [INTERRUPT] → Human reviews
                                                   ↓ approve
                                              Action executed
                                                   ↓ reject
                                              Agent re-plans
```

## Key LangGraph APIs
- `graph.compile(checkpointer=..., interrupt_before=["node_name"])`
- `graph.get_state(config)` — inspect state while paused
- `graph.update_state(config, values)` — optionally modify state before resume
- `graph.invoke(None, config)` — resume execution after human input

## Files
- `pattern.py` — reusable `HumanApprovalGate` helper
- `example.py` — email drafting agent with approval gate before send
- `test_pattern.py` — tests for approval / rejection / modification flows

## Tradeoffs
| Pro | Con |
|-----|-----|
| Prevents irreversible mistakes | Adds latency; requires human availability |
| Full audit trail via checkpointer | State must be persisted (can't use in-memory only) |
| Agent can incorporate feedback and retry | More complex graph topology |
