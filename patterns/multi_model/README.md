# Pattern: Multi-model Orchestration

## What It Is
Multi-model orchestration routes queries to the most appropriate model
based on task type, complexity, cost constraints, or required capabilities.
A supervisor/router model first classifies the query, then delegates to
a specialist model (or agent).

## Routing Strategies

| Strategy | How it works | Best for |
|----------|-------------|---------|
| **Task-based** | Classify query type → route to specialist | Math → calculator model, code → code model |
| **Complexity-based** | Simple → cheap model; Complex → powerful model | Cost optimisation |
| **Capability-based** | Route to model with required tool/skill | Routing to RAG vs ReAct vs Drafter |
| **Ensemble** | Run multiple models, vote or merge | High-stakes decisions |

## How It Works

```
User query
    ↓
Router LLM classifies intent
    ↓
    ├── "math"    → ReAct agent with calculator tools
    ├── "rag"     → RAG agent with document retrieval
    ├── "draft"   → Drafter agent
    └── "general" → General chat model
```

## Files
- `pattern.py` — `ModelRouter`, `SpecialistModel`, intent classifier
- `example.py` — 4-specialist orchestrator (math, code, creative, factual)
- `test_pattern.py` — tests for routing logic and specialist dispatch

## Tradeoffs
| Pro | Con |
|-----|-----|
| Right model for right task → better quality | Router adds one extra LLM call |
| Cost optimisation via complexity routing | Misclassification sends to wrong specialist |
| Modular — easy to add new specialists | More models = more infrastructure |
