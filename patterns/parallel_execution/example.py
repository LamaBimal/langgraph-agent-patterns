"""
example.py — Parallel Execution: Multi-source research agent

Runs three research branches in parallel:
  - branch_facts:    general facts about the topic (LLM call)
  - branch_pros:     pros/advantages (LLM call)
  - branch_cons:     cons/disadvantages (LLM call)

All three run simultaneously. A fan-in node synthesises the results.
Compare sequential (3 LLM calls back-to-back) vs parallel timing.

Run:
    python patterns/parallel_execution/example.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from parallel_pattern import build_parallel_graph, merge_results, run_parallel_tasks
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── LLM ───────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)


def _ask(system: str, query: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=query),
    ])
    return response.content


# ── Branch functions ───────────────────────────────────────────────────────
def branch_facts(state) -> str:
    return _ask("Give 3 key facts about the topic in bullet points.", state["query"])

def branch_pros(state) -> str:
    return _ask("List 3 main advantages or benefits in bullet points.", state["query"])

def branch_cons(state) -> str:
    return _ask("List 3 main disadvantages or risks in bullet points.", state["query"])

def fan_in(state) -> dict:
    """Synthesise all branch results into a structured report."""
    results = state.get("results", {})
    topic = state["query"]
    synthesis_prompt = f"""
Topic: {topic}

Facts:
{results.get('branch_facts', 'N/A')}

Pros:
{results.get('branch_pros', 'N/A')}

Cons:
{results.get('branch_cons', 'N/A')}

Write a balanced 2-sentence summary of the above.
"""
    summary = llm.invoke([
        SystemMessage(content="You are a research synthesiser."),
        HumanMessage(content=synthesis_prompt),
    ])
    merged = merge_results(state)
    return {**merged, "merged_output": summary.content}


# ── Graph ──────────────────────────────────────────────────────────────────
app = build_parallel_graph(
    branches={"branch_facts": branch_facts, "branch_pros": branch_pros, "branch_cons": branch_cons},
    fan_in_node=fan_in,
)


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== PARALLEL EXECUTION EXAMPLE =====\n")

    topic = input("Research topic (or press Enter for 'Remote Work'): ").strip()
    if not topic:
        topic = "Remote Work"

    # ── Parallel run ──────────────────────────────────────────────────────
    print(f"\nRunning 3 research branches in PARALLEL for: '{topic}'")
    t0 = time.monotonic()
    result = app.invoke({"query": topic, "results": {}, "errors": {}, "merged_output": ""})
    parallel_time = time.monotonic() - t0

    print(f"\n📊 SYNTHESIS:\n{result['merged_output']}")

    # ── Sequential comparison ──────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Running same 3 branches SEQUENTIALLY for comparison...")
    t1 = time.monotonic()
    _ = branch_facts({"query": topic})
    _ = branch_pros({"query": topic})
    _ = branch_cons({"query": topic})
    sequential_time = time.monotonic() - t1

    print(f"\n⏱  Timing comparison:")
    print(f"  Parallel:   {parallel_time:.2f}s")
    print(f"  Sequential: {sequential_time:.2f}s")
    speedup = sequential_time / max(parallel_time, 0.001)
    print(f"  Speedup:    {speedup:.1f}×")
