"""
pattern.py — Evaluation reusable building block.

Provides:
  - EvalScore: dataclass for a single evaluation result
  - EvalDataset: collection of (question, expected_answer, context) triples
  - LLMJudge: scores responses using an LLM as the evaluator
  - ReferenceScorer: compares against known correct answers
  - EvaluationReport: aggregates scores across a dataset

Usage:
    from patterns.evaluation.pattern import LLMJudge, EvalDataset, EvalScore

    judge = LLMJudge(llm=my_llm)
    score = judge.score(question="What is 2+2?", response="4", context="")
    print(score.overall)   # 0.0 – 1.0
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class EvalScore:
    """Result of evaluating a single response."""
    question: str
    response: str
    correctness: float = 0.0    # 0–1: factual accuracy
    relevance: float = 0.0      # 0–1: answers the question
    groundedness: float = 0.0   # 0–1: supported by context
    completeness: float = 0.0   # 0–1: covers all aspects
    reasoning: str = ""         # judge's explanation

    @property
    def overall(self) -> float:
        """Weighted average of all dimensions."""
        weights = {"correctness": 0.35, "relevance": 0.30,
                   "groundedness": 0.20, "completeness": 0.15}
        return (
            self.correctness * weights["correctness"] +
            self.relevance * weights["relevance"] +
            self.groundedness * weights["groundedness"] +
            self.completeness * weights["completeness"]
        )

    def __str__(self) -> str:
        return (
            f"Q: {self.question[:60]}\n"
            f"  Correctness:  {self.correctness:.2f}\n"
            f"  Relevance:    {self.relevance:.2f}\n"
            f"  Groundedness: {self.groundedness:.2f}\n"
            f"  Completeness: {self.completeness:.2f}\n"
            f"  Overall:      {self.overall:.2f}\n"
            f"  Reasoning:    {self.reasoning[:100]}"
        )


@dataclass
class EvalExample:
    """A single evaluation test case."""
    question: str
    expected_answer: str = ""
    context: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of evaluation examples."""
    name: str
    examples: list[EvalExample] = field(default_factory=list)

    def add(self, question: str, expected: str = "", context: str = "") -> None:
        self.examples.append(EvalExample(question, expected, context))

    def __len__(self) -> int:
        return len(self.examples)


# ── LLM-as-judge ──────────────────────────────────────────────────────────

JUDGE_PROMPT = """
You are an expert evaluator assessing the quality of an AI assistant's response.

Question: {question}
Context provided to the AI: {context}
AI Response: {response}
Expected answer (if available): {expected}

Rate the response on each dimension from 0.0 to 1.0:
- correctness: Is the response factually accurate?
- relevance: Does it directly answer the question?
- groundedness: Is it supported by the provided context? (1.0 if no context required)
- completeness: Does it cover all important aspects?

Respond with ONLY valid JSON in this exact format:
{{
  "correctness": 0.0,
  "relevance": 0.0,
  "groundedness": 0.0,
  "completeness": 0.0,
  "reasoning": "brief explanation"
}}
"""


class LLMJudge:
    """
    Uses an LLM to score agent responses on multiple quality dimensions.
    Works without labeled data — the LLM itself is the judge.
    """

    def __init__(self, llm, temperature: float = 0.1):
        # Use a low temperature for consistent scoring
        self.llm = llm

    def score(
        self,
        question: str,
        response: str,
        context: str = "",
        expected: str = "",
    ) -> EvalScore:
        """Score a single response."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            context=context or "No context provided",
            response=response,
            expected=expected or "Not provided",
        )

        try:
            raw = self.llm.invoke([
                SystemMessage(content="You are a precise evaluator. Respond only with valid JSON."),
                HumanMessage(content=prompt),
            ])
            content = raw.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            return EvalScore(
                question=question,
                response=response,
                correctness=float(scores.get("correctness", 0.0)),
                relevance=float(scores.get("relevance", 0.0)),
                groundedness=float(scores.get("groundedness", 1.0)),
                completeness=float(scores.get("completeness", 0.0)),
                reasoning=scores.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}. Returning neutral scores.")
            return EvalScore(
                question=question,
                response=response,
                correctness=0.5, relevance=0.5,
                groundedness=0.5, completeness=0.5,
                reasoning=f"Evaluation failed: {e}",
            )


# ── Reference scorer ───────────────────────────────────────────────────────

class ReferenceScorer:
    """
    Scores responses by comparing against known correct answers.
    Uses token overlap (F1) as the similarity metric.
    """

    def score(self, response: str, expected: str) -> float:
        """
        Compute token-level F1 overlap between response and expected answer.
        Returns 0.0–1.0.
        """
        if not expected:
            return 1.0  # No reference = can't penalise

        def tokenise(text: str) -> set[str]:
            return set(re.sub(r'[^\w\s]', '', text.lower()).split())

        resp_tokens = tokenise(response)
        exp_tokens = tokenise(expected)

        if not exp_tokens:
            return 1.0
        if not resp_tokens:
            return 0.0

        overlap = resp_tokens & exp_tokens
        precision = len(overlap) / len(resp_tokens)
        recall = len(overlap) / len(exp_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ── Evaluation report ──────────────────────────────────────────────────────

class EvaluationReport:
    """Aggregates EvalScores across a dataset."""

    def __init__(self, scores: list[EvalScore]):
        self.scores = scores

    @property
    def avg_overall(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.overall for s in self.scores) / len(self.scores)

    @property
    def avg_correctness(self) -> float:
        return sum(s.correctness for s in self.scores) / max(len(self.scores), 1)

    @property
    def avg_relevance(self) -> float:
        return sum(s.relevance for s in self.scores) / max(len(self.scores), 1)

    def print_report(self) -> None:
        print(f"\n{'='*55}")
        print(f"EVALUATION REPORT  ({len(self.scores)} examples)")
        print(f"{'='*55}")
        print(f"  Avg Overall:      {self.avg_overall:.3f}")
        print(f"  Avg Correctness:  {self.avg_correctness:.3f}")
        print(f"  Avg Relevance:    {self.avg_relevance:.3f}")
        print(f"{'─'*55}")
        for i, score in enumerate(self.scores):
            print(f"\n  [{i+1}] {score}")
        print(f"{'='*55}\n")

    def to_dict(self) -> dict:
        return {
            "total_examples": len(self.scores),
            "avg_overall": round(self.avg_overall, 3),
            "avg_correctness": round(self.avg_correctness, 3),
            "avg_relevance": round(self.avg_relevance, 3),
        }
