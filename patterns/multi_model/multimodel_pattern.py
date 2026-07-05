"""
pattern.py — Multi-model Orchestration reusable building block.

Provides:
  - IntentClassifier: uses an LLM to classify query intent
  - SpecialistAgent: wraps a model with a role-specific system prompt
  - ModelRouter: routes queries to the right specialist
  - build_orchestrator(): builds a full router → specialist graph

Usage:
    from patterns.multi_model.pattern import ModelRouter, SpecialistAgent, build_orchestrator

    router = ModelRouter(classifier_llm=llm, routes={
        "math": math_specialist,
        "creative": creative_specialist,
        "general": general_specialist,
    })
    response, route_used = router.route(query)
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Callable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    intent: str           # classified intent (e.g. "math", "creative", "general")
    confidence: float     # router confidence score 0–1
    response: str         # final response from specialist
    specialist_used: str  # which specialist handled the query


# ── Intent classifier ──────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """
Classify the user's query into exactly one of these categories: {categories}

Query: {query}

Respond with ONLY valid JSON:
{{"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}}
"""


class IntentClassifier:
    """
    Uses an LLM to classify query intent into predefined categories.
    Falls back to "general" on any failure.
    """

    def __init__(self, llm, categories: list[str], default: str = "general"):
        self.llm = llm
        self.categories = categories
        self.default = default

    def classify(self, query: str) -> tuple[str, float]:
        """
        Classify a query.

        Returns:
            (intent, confidence) tuple
        """
        prompt = CLASSIFIER_PROMPT.format(
            categories=", ".join(self.categories),
            query=query,
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a precise query classifier. Respond only with JSON."),
                HumanMessage(content=prompt),
            ])
            content = response.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                intent = data.get("intent", self.default)
                confidence = float(data.get("confidence", 0.5))

                # Validate intent is in known categories
                if intent not in self.categories:
                    logger.warning(f"Unknown intent '{intent}', falling back to '{self.default}'")
                    intent = self.default

                logger.info(f"Classified as '{intent}' (confidence: {confidence:.2f})")
                return intent, confidence
        except Exception as e:
            logger.warning(f"Classification failed: {e}, using default '{self.default}'")

        return self.default, 0.0


# ── Specialist agent ───────────────────────────────────────────────────────

@dataclass
class SpecialistAgent:
    """
    A model specialised for a particular task via a system prompt.

    Attributes:
        name:          identifier for this specialist
        llm:           the chat model to use
        system_prompt: role-specific instructions
        max_tokens:    override token limit for this specialist
    """
    name: str
    llm: object
    system_prompt: str
    max_tokens: int = 512

    def invoke(self, query: str, context: list[BaseMessage] = None) -> str:
        """Run the specialist on a query and return the response text."""
        messages = [SystemMessage(content=self.system_prompt)]
        if context:
            messages += context
        messages.append(HumanMessage(content=query))

        logger.info(f"  [specialist:{self.name}] processing query")
        response = self.llm.invoke(messages)
        return response.content


# ── Model router ───────────────────────────────────────────────────────────

class ModelRouter:
    """
    Routes queries to the appropriate specialist agent.

    Args:
        classifier_llm: LLM used to classify query intent
        routes:         {intent_name: SpecialistAgent} mapping
        default_route:  fallback intent name if classification fails
        confidence_threshold: below this, use default route
    """

    def __init__(
        self,
        classifier_llm,
        routes: dict[str, SpecialistAgent],
        default_route: str = "general",
        confidence_threshold: float = 0.4,
    ):
        self.classifier = IntentClassifier(
            llm=classifier_llm,
            categories=list(routes.keys()),
            default=default_route,
        )
        self.routes = routes
        self.default_route = default_route
        self.confidence_threshold = confidence_threshold

    def route(self, query: str, context: list = None) -> tuple[str, str]:
        """
        Classify the query and dispatch to the appropriate specialist.

        Returns:
            (response_text, specialist_name_used)
        """
        intent, confidence = self.classifier.classify(query)

        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({confidence:.2f}), using default route '{self.default_route}'")
            intent = self.default_route

        specialist = self.routes.get(intent, self.routes.get(self.default_route))
        if specialist is None:
            return "No specialist available for this query.", "none"

        response = specialist.invoke(query, context)
        return response, specialist.name


# ── Graph builder ──────────────────────────────────────────────────────────

def build_orchestrator(router: ModelRouter) -> object:
    """
    Build a LangGraph orchestrator with routing built in.

    Graph flow:
        START → classify → route_to_specialist → END

    Args:
        router: ModelRouter instance with specialists configured

    Returns:
        Compiled StateGraph
    """

    def classify_node(state: OrchestratorState) -> OrchestratorState:
        query = state.get("query") or (
            str(state["messages"][-1].content) if state.get("messages") else ""
        )
        intent, confidence = router.classifier.classify(query)
        if confidence < router.confidence_threshold:
            intent = router.default_route
        print(f"\n  [router] Intent: {intent} (confidence: {confidence:.2f})")
        return {"intent": intent, "confidence": confidence, "query": query}

    def dispatch_node(state: OrchestratorState) -> OrchestratorState:
        intent = state.get("intent", router.default_route)
        specialist = router.routes.get(intent, router.routes.get(router.default_route))

        if specialist is None:
            response = "No specialist available."
            name = "none"
        else:
            response = specialist.invoke(state["query"], list(state.get("messages", [])))
            name = specialist.name

        print(f"  [specialist:{name}] responded ({len(response)} chars)")
        return {
            "response": response,
            "specialist_used": name,
            "messages": [AIMessage(content=response)],
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("classify", classify_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "dispatch")
    graph.add_edge("dispatch", END)

    return graph.compile()
