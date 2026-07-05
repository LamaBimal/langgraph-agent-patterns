# Pattern: Parallel Execution

## What It Is
Parallel execution runs multiple graph nodes simultaneously (fan-out),
then aggregates their results (fan-in). This is the agent equivalent
of multi-threading: independent tasks are completed concurrently,
dramatically reducing total latency.

## When to Use
- Multiple independent tool calls (search + calculate + fetch)
- RAG with multiple retrievers (different document sources)
- Ensemble responses: run same query on multiple models, merge results
- Data enrichment: enrich a record from multiple APIs simultaneously

## LangGraph Approach
LangGraph supports parallel nodes natively. Any node with multiple
incoming edges from the same source runs in parallel. State updates
from parallel branches are merged using the reducer defined in state.

```python
# Fan-out: both branch_a and branch_b run in parallel
graph.add_edge("start_node", "branch_a")
graph.add_edge("start_node", "branch_b")

# Fan-in: merge runs after both branches complete
graph.add_edge("branch_a", "merge")
graph.add_edge("branch_b", "merge")
```

## Files
- `pattern.py` — `ParallelBranch`, fan-out/fan-in helpers, result merger
- `example.py` — parallel search + calculation agent
- `test_pattern.py` — tests for parallel execution and result merging

## Tradeoffs
| Pro | Con |
|-----|-----|
| 2–5× speedup on independent tasks | Added complexity in state merging |
| Scales naturally with more branches | One slow branch blocks the fan-in |
| No external threading required | Harder to debug than sequential flow |
