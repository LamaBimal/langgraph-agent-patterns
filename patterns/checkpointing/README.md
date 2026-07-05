# Pattern: Checkpointing

## What It Is
Checkpointing persists the full graph state after every node execution so
that a workflow can be paused, resumed, replayed, or recovered without
starting over. LangGraph supports multiple checkpointer backends.

## When to Use
- Long-running workflows that may span minutes or hours
- Human-in-the-loop flows where the process waits for external input
- Resumable workflows after a crash or restart
- Branching / time-travel: re-run from any prior checkpoint
- Audit trails: inspect exactly what the agent did at each step

## Backends Supported

| Backend | Use case |
|---------|----------|
| `MemorySaver` | Development and testing (in-process, not persistent) |
| `SqliteSaver` | Single-process production, local file persistence |
| `PostgresSaver` | Multi-process / distributed production |
| Custom | Any key-value store (Redis, DynamoDB, etc.) |

## How It Works

```
graph.compile(checkpointer=SqliteSaver.from_conn_string("db.sqlite"))

# Each invoke is scoped to a thread_id
config = {"configurable": {"thread_id": "user-123-session-1"}}
graph.invoke(input, config=config)

# Inspect history
for state in graph.get_state_history(config):
    print(state.values, state.next)

# Resume from latest checkpoint
graph.invoke(None, config=config)

# Time-travel: resume from specific checkpoint
graph.invoke(None, config={**config, "checkpoint_id": past_checkpoint_id})
```

## Files
- `pattern.py` — `CheckpointerFactory`, checkpoint helpers, history explorer
- `example.py` — multi-step research agent with SQLite checkpointing and resume
- `test_pattern.py` — tests for checkpoint creation, resume, history inspection

## Tradeoffs
| Pro | Con |
|-----|-----|
| Full crash recovery | Storage cost grows with history |
| Time-travel debugging | SQLite not suitable for concurrent writes |
| Enables HITL without losing state | Checkpoint schema tied to LangGraph version |
