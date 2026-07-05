"""Tests for memory pattern."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from memory_pattern import ShortTermMemory, LongTermMemory, EpisodicMemory, MemoryManager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ── ShortTermMemory ────────────────────────────────────────────────────────

def test_trim_keeps_recent_messages():
    mem = ShortTermMemory(window_size=3)
    msgs = [HumanMessage(content=str(i)) for i in range(10)]
    trimmed = mem.trim(msgs)
    assert len(trimmed) == 3
    assert trimmed[-1].content == "9"


def test_trim_preserves_system_message():
    mem = ShortTermMemory(window_size=2)
    msgs = [SystemMessage(content="sys")] + [HumanMessage(content=str(i)) for i in range(5)]
    trimmed = mem.trim(msgs)
    assert any(isinstance(m, SystemMessage) for m in trimmed)
    assert trimmed[0].content == "sys"


def test_no_trim_when_under_window():
    mem = ShortTermMemory(window_size=20)
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    trimmed = mem.trim(msgs)
    assert len(trimmed) == 2


def test_token_estimate():
    mem = ShortTermMemory()
    msgs = [HumanMessage(content="a" * 400)]  # 400 chars ≈ 100 tokens
    assert mem.token_estimate(msgs) == 100


# ── LongTermMemory ─────────────────────────────────────────────────────────

def test_store_and_retrieve(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    lt.store("user_name", "Bimal")
    assert lt.retrieve("user_name") == "Bimal"


def test_retrieve_missing_returns_default(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    assert lt.retrieve("nonexistent", "default") == "default"


def test_search_finds_by_key(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    lt.store("user_city", "Kathmandu")
    results = lt.search("city")
    assert any("Kathmandu" in str(v) for _, v in results)


def test_delete(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    lt.store("temp_key", "value")
    assert lt.delete("temp_key") is True
    assert lt.retrieve("temp_key") is None


def test_persist_across_instances(tmp_path):
    path = str(tmp_path / "lt.json")
    lt1 = LongTermMemory(path)
    lt1.store("persistent_key", "persistent_value")

    lt2 = LongTermMemory(path)
    assert lt2.retrieve("persistent_key") == "persistent_value"


# ── EpisodicMemory ─────────────────────────────────────────────────────────

def test_save_and_recall_episode(tmp_path):
    ep = EpisodicMemory(str(tmp_path / "ep.json"))
    ep.save_episode("s1", "User asked about Python", topics=["python"])
    results = ep.recall("python")
    assert len(results) == 1
    assert "Python" in results[0]["summary"]


def test_recent_episodes(tmp_path):
    ep = EpisodicMemory(str(tmp_path / "ep.json"))
    for i in range(5):
        ep.save_episode(f"s{i}", f"Session {i} summary")
    recent = ep.recent(3)
    assert len(recent) == 3
    assert recent[-1]["session_id"] == "s4"


# ── MemoryManager ──────────────────────────────────────────────────────────

def test_extract_name_fact(tmp_path):
    mgr = MemoryManager(
        lt_storage=str(tmp_path / "lt.json"),
        ep_storage=str(tmp_path / "ep.json"),
    )
    mgr.extract_and_store_facts("My name is Alice")
    assert mgr.long_term.retrieve("user_name") == "alice"
