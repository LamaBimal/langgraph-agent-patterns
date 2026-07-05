# Pattern: Evaluation

## What It Is
Evaluation measures whether your agent is actually producing correct,
relevant, and safe responses. It closes the feedback loop between
development and production by providing quantitative quality scores.

## Evaluation Dimensions

| Dimension | Question | Score range |
|-----------|----------|-------------|
| **Correctness** | Is the answer factually right? | 0–1 |
| **Relevance** | Does it answer the actual question? | 0–1 |
| **Groundedness** | Is it supported by retrieved context? | 0–1 (RAG) |
| **Completeness** | Does it cover all aspects? | 0–1 |
| **Safety** | Does it contain harmful content? | 0–1 |

## When to Use
- Before deploying a new agent or RAG pipeline to production
- Regression testing after changing prompts or models
- A/B comparing two model versions
- Monitoring production quality over time (continuous eval)

## Approaches

1. **LLM-as-judge** — Use a strong LLM to score responses (cheap, flexible)
2. **Reference-based** — Compare against golden answers (precise, needs labeled data)
3. **RAGAS** — Specialised RAG evaluation metrics (requires `ragas` package)

## Files
- `pattern.py` — `Evaluator`, `EvalScore`, LLM-as-judge scorer, reference scorer
- `example.py` — evaluate a RAG agent against a test dataset
- `test_pattern.py` — tests for scoring functions and aggregation

## Tradeoffs
| Pro | Con |
|-----|-----|
| Objective quality measurement | LLM-as-judge has its own biases |
| Catches regressions early | Labeled datasets are expensive to create |
| Works without ground truth (LLM judge) | Scores can be inconsistent across runs |
