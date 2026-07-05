"""
pattern.py — Memory (Short-term, Long-term, Episodic) reusable building block.

Provides:
  - ShortTermMemory: sliding window of recent messages
  - LongTermMemory: persistent key-value fact store with fuzzy search
  - EpisodicMemory: session summarisation and cross-session recall
  - MemoryManager: orchestrates all three memory types

Usage:
    from patterns.memory.pattern import MemoryManager, ShortTermMemory

    mem = MemoryManager(window_size=10)
    trimmed_messages = mem.short_term.trim(messages)
    mem.long_term.store("user_name", "Bimal")
    facts = mem.long_term.search("name")
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── Short-term memory ──────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Keeps the N most recent messages to stay within the model's context window.
    Always preserves the SystemMessage if present.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Return messages trimmed to window_size.
        SystemMessages are always kept; HumanMessage/AIMessage pairs trimmed.
        """
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) > self.window_size:
            logger.debug(f"Trimming {len(other_msgs) - self.window_size} old messages")
            other_msgs = other_msgs[-self.window_size:]

        return system_msgs + other_msgs

    def token_estimate(self, messages: list[BaseMessage]) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        total_chars = sum(len(str(m.content)) for m in messages)
        return total_chars // 4


# ── Long-term memory ───────────────────────────────────────────────────────

class LongTermMemory:
    """
    Simple persistent key-value fact store backed by a JSON file.
    In production, replace with a vector store (Chroma, Pinecone, etc.)
    for semantic similarity search.
    """

    def __init__(self, storage_path: str = "./long_term_memory.json"):
        self.storage_path = Path(storage_path)
        self._facts: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._facts, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def store(self, key: str, value: Any) -> None:
        """Store a fact."""
        self._facts[key] = value
        self._save()
        logger.debug(f"Stored: {key} = {value}")

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact by exact key."""
        return self._facts.get(key, default)

    def search(self, query: str) -> list[tuple[str, Any]]:
        """
        Search facts by substring match on keys and string values.
        In production, replace with vector similarity search.
        """
        query_lower = query.lower()
        results = []
        for k, v in self._facts.items():
            if query_lower in k.lower() or query_lower in str(v).lower():
                results.append((k, v))
        return results

    def all_facts(self) -> dict:
        return dict(self._facts)

    def delete(self, key: str) -> bool:
        if key in self._facts:
            del self._facts[key]
            self._save()
            return True
        return False

    def format_for_prompt(self, facts: list[tuple] = None) -> str:
        """Format facts as a context string for injection into a system prompt."""
        items = facts or list(self._facts.items())
        if not items:
            return ""
        lines = ["Known facts about the user:"]
        for k, v in items:
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


# ── Episodic memory ────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Stores summaries of past sessions for cross-session recall.
    Each episode = {session_id, summary, timestamp, key_topics}
    """

    def __init__(self, storage_path: str = "./episodic_memory.json"):
        self.storage_path = Path(storage_path)
        self._episodes: list[dict] = self._load()

    def _load(self) -> list:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._episodes, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def save_episode(self, session_id: str, summary: str, topics: list[str] = None) -> None:
        """Save a session summary as an episode."""
        import datetime
        episode = {
            "session_id": session_id,
            "summary": summary,
            "topics": topics or [],
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        self._episodes.append(episode)
        self._save()
        logger.info(f"Saved episode for session: {session_id}")

    def recall(self, query: str, max_episodes: int = 3) -> list[dict]:
        """Retrieve the most relevant past episodes for a query."""
        query_lower = query.lower()
        scored = []
        for ep in self._episodes:
            score = 0
            if query_lower in ep["summary"].lower():
                score += 2
            for topic in ep.get("topics", []):
                if query_lower in topic.lower():
                    score += 1
            if score > 0:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:max_episodes]]

    def recent(self, n: int = 3) -> list[dict]:
        """Return the N most recent episodes."""
        return self._episodes[-n:]

    def format_for_prompt(self, episodes: list[dict] = None) -> str:
        """Format episodes as context for a system prompt."""
        items = episodes or self.recent(3)
        if not items:
            return ""
        lines = ["Previous conversation context:"]
        for ep in items:
            lines.append(f"  [{ep.get('timestamp', 'unknown')}] {ep['summary']}")
        return "\n".join(lines)


# ── Memory manager ─────────────────────────────────────────────────────────

class MemoryManager:
    """
    Orchestrates all three memory types.

    Usage:
        mem = MemoryManager(
            window_size=20,
            lt_storage="./memory/long_term.json",
            ep_storage="./memory/episodes.json",
        )
        messages = mem.short_term.trim(messages)
        context = mem.build_context_prompt("user question")
    """

    def __init__(
        self,
        window_size: int = 20,
        lt_storage: str = "./long_term_memory.json",
        ep_storage: str = "./episodic_memory.json",
    ):
        self.short_term = ShortTermMemory(window_size=window_size)
        self.long_term = LongTermMemory(storage_path=lt_storage)
        self.episodic = EpisodicMemory(storage_path=ep_storage)

    def build_context_prompt(self, query: str) -> str:
        """
        Build a context string to prepend to the system prompt,
        combining relevant long-term facts and episodic memories.
        """
        parts = []

        facts = self.long_term.search(query)
        if facts:
            parts.append(self.long_term.format_for_prompt(facts))

        episodes = self.episodic.recall(query)
        if episodes:
            parts.append(self.episodic.format_for_prompt(episodes))

        return "\n\n".join(parts)

    def extract_and_store_facts(self, text: str) -> None:
        """
        Simple rule-based fact extraction from conversation text.
        In production, use an LLM to extract structured facts.
        """
        patterns = {
            "user_name": r"my name is (\w+)",
            "user_location": r"i(?:'m| am) from ([A-Za-z\s]+)",
            "user_preference": r"i (?:like|prefer|love) ([^.!?]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                value = match.group(1).strip()
                self.long_term.store(key, value)
                logger.info(f"Extracted fact: {key} = {value}")
