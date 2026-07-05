"""Tests for evaluation pattern."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from evaluation_pattern import EvalScore, EvalDataset, ReferenceScorer, EvaluationReport


def test_eval_score_overall_weighted():
    s = EvalScore("q", "r", correctness=1.0, relevance=1.0, groundedness=1.0, completeness=1.0)
    assert abs(s.overall - 1.0) < 0.001


def test_eval_score_overall_zero():
    s = EvalScore("q", "r", correctness=0.0, relevance=0.0, groundedness=0.0, completeness=0.0)
    assert s.overall == 0.0


def test_reference_scorer_exact_match():
    rs = ReferenceScorer()
    score = rs.score("Paris is the capital of France", "Paris is the capital of France")
    assert score == 1.0


def test_reference_scorer_partial_match():
    rs = ReferenceScorer()
    score = rs.score("Paris", "Paris is the capital of France")
    assert 0.0 < score < 1.0


def test_reference_scorer_no_overlap():
    rs = ReferenceScorer()
    # Use completely distinct vocabulary to guarantee zero overlap
    score = rs.score("zebra elephant giraffe", "piano violin trumpet")
    assert score == 0.0


def test_reference_scorer_empty_expected():
    rs = ReferenceScorer()
    score = rs.score("anything", "")
    assert score == 1.0  # no reference = no penalty


def test_eval_dataset_add():
    ds = EvalDataset("test")
    ds.add("Q1", expected="A1")
    ds.add("Q2")
    assert len(ds) == 2
    assert ds.examples[0].expected_answer == "A1"


def test_evaluation_report_averages():
    scores = [
        EvalScore("q1", "r1", correctness=1.0, relevance=1.0, groundedness=1.0, completeness=1.0),
        EvalScore("q2", "r2", correctness=0.0, relevance=0.0, groundedness=0.0, completeness=0.0),
    ]
    report = EvaluationReport(scores)
    assert abs(report.avg_overall - 0.5) < 0.01
    assert abs(report.avg_correctness - 0.5) < 0.01


def test_evaluation_report_empty():
    report = EvaluationReport([])
    assert report.avg_overall == 0.0


def test_report_to_dict():
    scores = [EvalScore("q", "r", correctness=0.8, relevance=0.9, groundedness=0.7, completeness=0.6)]
    report = EvaluationReport(scores)
    d = report.to_dict()
    assert "avg_overall" in d
    assert d["total_examples"] == 1
