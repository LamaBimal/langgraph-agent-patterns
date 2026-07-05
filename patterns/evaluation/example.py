"""
example.py — Evaluation: LLM-as-judge on a QA dataset

Runs a simple QA agent against a 5-question test dataset and scores
each response using the LLM-as-judge approach. Prints an evaluation
report with per-dimension scores and overall quality.

Run:
    python patterns/evaluation/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from evaluation_pattern import LLMJudge, EvalDataset, EvaluationReport, EvalScore, ReferenceScorer
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── Models ─────────────────────────────────────────────────────────────────
agent_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

judge_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.1,  # low temp for consistent scoring
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)


# ── Test dataset ───────────────────────────────────────────────────────────
dataset = EvalDataset(name="general-qa-v1")
dataset.add("What is the capital of France?", expected="Paris")
dataset.add("What does CPU stand for?", expected="Central Processing Unit")
dataset.add("What language is LangGraph written in?", expected="Python")
dataset.add("What is 15 multiplied by 7?", expected="105")
dataset.add(
    "Explain what a neural network is.",
    expected="A neural network is a computational model inspired by the human brain, "
             "consisting of layers of interconnected nodes that learn patterns from data.",
)


# ── Agent (simple QA) ──────────────────────────────────────────────────────
def run_agent(question: str) -> str:
    response = agent_llm.invoke([
        SystemMessage(content="Answer the following question concisely and accurately."),
        HumanMessage(content=question),
    ])
    return response.content


# ── Evaluation loop ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== EVALUATION EXAMPLE =====")
    print(f"Dataset: {dataset.name} | {len(dataset)} examples\n")

    judge = LLMJudge(llm=judge_llm)
    ref_scorer = ReferenceScorer()
    all_scores: list[EvalScore] = []

    for i, example in enumerate(dataset.examples):
        print(f"[{i+1}/{len(dataset)}] Q: {example.question[:60]}")
        response = run_agent(example.question)
        print(f"  A: {response[:80]}...")

        # LLM-as-judge scoring
        score = judge.score(
            question=example.question,
            response=response,
            expected=example.expected_answer,
        )

        # Boost correctness with reference F1 if we have an expected answer
        if example.expected_answer:
            ref_score = ref_scorer.score(response, example.expected_answer)
            score.correctness = (score.correctness + ref_score) / 2

        print(f"  Score: {score.overall:.2f}\n")
        all_scores.append(score)

    report = EvaluationReport(all_scores)
    report.print_report()
