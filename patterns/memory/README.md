# Pattern: Memory

## What It Is
Memory gives agents the ability to recall information across turns, sessions,
and even users. This pattern covers three memory types that work together:

| Type | Scope | Storage | Use case |
|------|-------|---------|----------|
| **Short-term** | Current session | In-state message list | Multi-turn chat context |
| **Long-term** | Across sessions | Vector store / KV store | User preferences, facts |
| **Episodic** | Past interactions | Summarised checkpoints | Recall "last time we discussed..." |

## When to Use
- Personalised assistants that remember user preferences
- Support agents that recall previous tickets for a customer
- Research agents that accumulate knowledge over multiple sessions
- Any multi-session agent where context beyond the current window matters

## How It Works

```
Short-term: messages list in state (trimmed to window size)
    ↓ on session end
Long-term: extract key facts → embed → store in vector DB
    ↓ on new session
Retrieval: embed query → similarity search → inject as context
    ↓
Episodic: summarise previous session → prepend to new session
```

## Files
- `pattern.py` — `ShortTermMemory`, `LongTermMemory`, `EpisodicMemory`
- `example.py` — personal assistant that remembers facts across sessions
- `test_pattern.py` — tests for memory store, retrieval, summarisation

## Tradeoffs
| Pro | Con |
|-----|-----|
| Personalised, context-aware responses | Vector store adds infrastructure |
| Scales beyond context window | Memory may surface stale/incorrect facts |
| Episodic recall feels natural | Summarisation loses detail |
