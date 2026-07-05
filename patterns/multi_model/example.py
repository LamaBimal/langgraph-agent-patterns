"""
example.py — Multi-model Orchestration: 4-specialist router

Routes queries to one of four specialists:
  - math:     solve calculations step-by-step
  - code:     write and explain Python code
  - creative: storytelling and creative writing
  - general:  factual Q&A and general knowledge

The router LLM classifies intent first, then the specialist responds.

Run:
    python patterns/multi_model/example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from multimodel_pattern import SpecialistAgent, ModelRouter, build_orchestrator
from config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_VALIDATE_ON_INIT


# ── Models ─────────────────────────────────────────────────────────────────
# In production, use different model sizes per specialist
# (e.g. smaller model for routing, larger for complex tasks)
base_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=OLLAMA_TEMPERATURE,
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)

router_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.1,  # low temp for consistent classification
    validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
)


# ── Specialists ────────────────────────────────────────────────────────────
math_specialist = SpecialistAgent(
    name="math",
    llm=base_llm,
    system_prompt=(
        "You are a mathematics expert. Solve problems step-by-step, "
        "showing your working clearly. Use proper notation."
    ),
)

code_specialist = SpecialistAgent(
    name="code",
    llm=base_llm,
    system_prompt=(
        "You are an expert Python developer. Write clean, well-commented code. "
        "Always include a brief explanation of how your solution works."
    ),
)

creative_specialist = SpecialistAgent(
    name="creative",
    llm=ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.9,  # higher temperature for creativity
        validate_model_on_init=OLLAMA_VALIDATE_ON_INIT,
    ),
    system_prompt=(
        "You are a creative writer with a vivid imagination. "
        "Craft engaging, imaginative responses with descriptive language."
    ),
)

general_specialist = SpecialistAgent(
    name="general",
    llm=base_llm,
    system_prompt=(
        "You are a knowledgeable and helpful assistant. "
        "Provide accurate, concise, well-structured answers."
    ),
)


# ── Router and orchestrator ────────────────────────────────────────────────
router = ModelRouter(
    classifier_llm=router_llm,
    routes={
        "math": math_specialist,
        "code": code_specialist,
        "creative": creative_specialist,
        "general": general_specialist,
    },
    default_route="general",
    confidence_threshold=0.35,
)

app = build_orchestrator(router)


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n===== MULTI-MODEL ORCHESTRATION =====")
    print("Specialists: math | code | creative | general")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent": "",
            "confidence": 0.0,
            "response": "",
            "specialist_used": "",
        })

        print(f"\n[{result['specialist_used']}]: {result['response']}\n")
