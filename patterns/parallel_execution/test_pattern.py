"""Tests for parallel_execution pattern."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from parallel_pattern import make_branch, build_parallel_graph, merge_results, run_parallel_tasks


def test_make_branch_stores_result():
    def my_fn(state):
        return "result_value"
    branch = make_branch("my_branch", my_fn)
    result = branch({"query": "test", "results": {}, "errors": {}})
    assert result["results"]["my_branch"] == "result_value"


def test_make_branch_stores_error_on_failure():
    def failing_fn(state):
        raise ValueError("simulated failure")
    branch = make_branch("bad_branch", failing_fn)
    result = branch({"query": "test", "results": {}, "errors": {}})
    assert "bad_branch" in result["errors"]
    assert "simulated failure" in result["errors"]["bad_branch"]


def test_merge_results_combines_all():
    state = {
        "results": {"branch_a": "Result A", "branch_b": "Result B"},
        "errors": {},
        "query": "test",
        "merged_output": "",
    }
    merged = merge_results(state)
    assert "Result A" in merged["merged_output"]
    assert "Result B" in merged["merged_output"]


def test_merge_results_includes_errors():
    state = {
        "results": {},
        "errors": {"branch_c": "timeout"},
        "query": "test",
        "merged_output": "",
    }
    merged = merge_results(state)
    assert "ERROR" in merged["merged_output"]
    assert "timeout" in merged["merged_output"]


def test_build_parallel_graph_executes_branches():
    def branch_a(state):
        return "A done"
    def branch_b(state):
        return "B done"

    app = build_parallel_graph(
        branches={"branch_a": branch_a, "branch_b": branch_b},
    )
    result = app.invoke({"query": "test", "results": {}, "errors": {}, "merged_output": ""})
    assert result["results"].get("branch_a") == "A done"
    assert result["results"].get("branch_b") == "B done"


def test_run_parallel_tasks_returns_all():
    tasks = {
        "task1": lambda: "result1",
        "task2": lambda: "result2",
        "task3": lambda: "result3",
    }
    results = run_parallel_tasks(tasks)
    assert results["task1"] == "result1"
    assert results["task2"] == "result2"
    assert results["task3"] == "result3"


def test_run_parallel_tasks_captures_exceptions():
    tasks = {
        "good": lambda: "ok",
        "bad": lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    }
    results = run_parallel_tasks(tasks)
    assert results["good"] == "ok"
    assert isinstance(results["bad"], Exception)


def test_parallel_faster_than_sequential():
    """Parallel tasks should finish faster than sum of individual durations."""
    def slow_task():
        time.sleep(0.1)
        return "done"

    tasks = {f"t{i}": slow_task for i in range(3)}

    start = time.monotonic()
    run_parallel_tasks(tasks)
    elapsed = time.monotonic() - start

    # 3 × 0.1s tasks in parallel should finish well under 0.3s
    assert elapsed < 0.25, f"Expected < 0.25s but got {elapsed:.2f}s"
