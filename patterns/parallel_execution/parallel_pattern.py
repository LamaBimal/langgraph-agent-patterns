"""
pattern.py — Parallel Execution reusable building block.

Provides:
  - ParallelState: TypedDict with a results dict for collecting branch outputs
  - build_parallel_graph(): fan-out → parallel branches → fan-in
  - merge_results(): combiner function for parallel branch outputs
  - run_parallel_tasks(): run arbitrary callables in parallel via ThreadPoolExecutor

Usage:
    from patterns.parallel_execution.pattern import build_parallel_graph, ParallelState
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Callable, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)


# ── Dict merge reducer ─────────────────────────────────────────────────────

def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts — used as reducer for parallel state fields."""
    return {**a, **b}


# ── State ──────────────────────────────────────────────────────────────────

class ParallelState(TypedDict):
    """
    Base state for parallel execution patterns.

    results and errors use merge_dicts as reducer so parallel branches
    can each write their own key without clobbering each other.
    """
    query: str
    results: Annotated[dict, merge_dicts]   # populated by parallel branches
    errors: Annotated[dict, merge_dicts]    # populated on branch failure
    merged_output: str                      # final merged result from fan-in


# ── Branch wrapper ─────────────────────────────────────────────────────────

def make_branch(name: str, fn: Callable) -> Callable:
    """
    Wrap a function as a named parallel branch node.

    The function receives the state and returns a result value.
    The result is stored under state["results"][name].

    Args:
        name: unique branch identifier (used as results dict key)
        fn:   callable(state) → any result value

    Returns:
        LangGraph-compatible node function
    """
    def branch_node(state: ParallelState) -> dict:
        start = time.monotonic()
        try:
            result = fn(state)
            latency = (time.monotonic() - start) * 1000
            logger.info(f"  [parallel] {name} completed in {latency:.0f}ms")
            results = dict(state.get("results", {}))
            results[name] = result
            return {"results": results}
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.error(f"  [parallel] {name} failed in {latency:.0f}ms: {e}")
            errors = dict(state.get("errors", {}))
            errors[name] = str(e)
            return {"errors": errors}

    branch_node.__name__ = name
    return branch_node


# ── Graph builder ──────────────────────────────────────────────────────────

def build_parallel_graph(
    branches: dict[str, Callable],
    fan_out_node: Callable = None,
    fan_in_node: Callable = None,
) -> "compiled graph":
    """
    Build a fan-out → parallel branches → fan-in graph.

    Args:
        branches:      {branch_name: fn(state) → result} dict
        fan_out_node:  optional node that runs before branches (prepares query)
                       If None, branches start directly from START
        fan_in_node:   optional node that merges results
                       If None, graph ends after branches complete

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(ParallelState)

    # Add optional fan-out node
    if fan_out_node:
        graph.add_node("fan_out", fan_out_node)
        graph.add_edge(START, "fan_out")
        source = "fan_out"
    else:
        source = START

    # Add parallel branch nodes
    branch_names = []
    for name, fn in branches.items():
        node = make_branch(name, fn)
        graph.add_node(name, node)
        graph.add_edge(source, name)
        branch_names.append(name)

    # Add fan-in node
    if fan_in_node:
        graph.add_node("fan_in", fan_in_node)
        for name in branch_names:
            graph.add_edge(name, "fan_in")
        graph.add_edge("fan_in", END)
    else:
        for name in branch_names:
            graph.add_edge(name, END)

    return graph.compile()


# ── Fan-in merger ──────────────────────────────────────────────────────────

def merge_results(
    state: ParallelState,
    separator: str = "\n\n---\n\n",
    include_labels: bool = True,
) -> dict:
    """
    Default fan-in node: concatenates all branch results into merged_output.

    Args:
        separator: string between branch results
        include_labels: prefix each result with the branch name
    """
    results = state.get("results", {})
    errors = state.get("errors", {})

    parts = []
    for name, result in sorted(results.items()):
        if include_labels:
            parts.append(f"[{name}]\n{result}")
        else:
            parts.append(str(result))

    for name, error in errors.items():
        parts.append(f"[{name} - ERROR]\n{error}")

    merged = separator.join(parts)
    logger.info(f"Merged {len(results)} branch results, {len(errors)} errors")
    return {"merged_output": merged}


# ── Thread pool parallel execution (outside LangGraph) ────────────────────

def run_parallel_tasks(
    tasks: dict[str, Callable],
    timeout: float = 30.0,
) -> dict[str, any]:
    """
    Run arbitrary callables in parallel using ThreadPoolExecutor.
    Returns {task_name: result_or_exception} dict.

    Use this when you need parallelism outside a LangGraph graph
    (e.g. inside a single node that calls multiple APIs).

    Args:
        tasks:   {name: callable()} dict — callables take no arguments
        timeout: max seconds to wait for all tasks

    Returns:
        {name: result} or {name: Exception} on failure
    """
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = e
                logger.error(f"Task '{name}' failed: {e}")
    return results
