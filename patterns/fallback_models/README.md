# Pattern: Fallback Models

## What It Is
Fallback models chain multiple LLMs so that if the primary model is
unavailable, overloaded, or produces an unacceptable response, the agent
automatically falls back to a secondary (and tertiary) model.

## When to Use
- Production systems where uptime matters more than using a specific model
- Cost optimisation: try a cheaper model first, fall back to a more capable one
- Multi-provider resilience: OpenAI down → Anthropic → local Ollama
- Quality gating: if response confidence is too low, escalate to stronger model

## How It Works

```
Try primary model
    ↓ success → use response
    ↓ fail / low quality
Try secondary model
    ↓ success → use response
    ↓ fail
Try tertiary / final fallback
    ↓
Return best available response
```

## Key Concepts
- `ModelChain`: ordered list of (model, condition) pairs
- Condition can be: exception, empty response, quality score below threshold
- Each model attempt is logged for observability
- Final fallback can be a static response if all models fail

## Files
- `pattern.py` — `ModelChain`, `FallbackConfig`, quality checker
- `example.py` — agent with 3-tier Ollama model fallback chain
- `test_pattern.py` — tests for chain routing and quality gating

## Tradeoffs
| Pro | Con |
|-----|-----|
| High availability | Latency increases when falling back |
| Cost-optimised (cheap first) | Inconsistent response quality across models |
| Provider diversity reduces single point of failure | More complex configuration |
