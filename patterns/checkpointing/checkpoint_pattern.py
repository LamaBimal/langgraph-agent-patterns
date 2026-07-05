"""
pattern.py — Checkpointing reusable building block.

Provides:
  - CheckpointerFactory: creates the right checkpointer for the environment
  - CheckpointExplorer: utilities for inspecting and replaying history
  - thread_config(): builds the LangGraph config dict for a thread

Usage:
    from patterns.checkpointing.pattern import CheckpointerFactory, thread_config

    checkpointer = CheckpointerFactory.sqlite("./checkpoints.db")
    app = graph.compile(checkpointer=checkpointer)
    app.invoke(state, config=thread_config("user-123"))
"""

import os
import logging
from pathlib import Path
from typing import Any, Generator
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ── Thread config helper ───────────────────────────────────────────────────

def thread_config(thread_id: str, **extra) -> dict:
    """
    Build the LangGraph config dict for a specific thread.

    Args:
        thread_id: unique identifier for this conversation/session
        **extra: additional configurable keys

    Returns:
        {"configurable": {"thread_id": thread_id, ...}}
    """
    return {"configurable": {"thread_id": str(thread_id), **extra}}


# ── Factory ────────────────────────────────────────────────────────────────

class CheckpointerFactory:
    """
    Creates the appropriate checkpointer based on environment and config.
    """

    @staticmethod
    def memory() -> MemorySaver:
        """
        In-process memory checkpointer.
        Fast, zero-config — but state is lost on process exit.
        Use for: tests, development, demos.
        """
        logger.info("Using MemorySaver (in-process, non-persistent)")
        return MemorySaver()

    @staticmethod
    def sqlite(db_path: str = "./checkpoints.db"):
        """
        SQLite-backed checkpointer for single-process persistence.
        State survives process restarts.
        Use for: local production, single-user deployments.

        Requires: langgraph-checkpoint-sqlite
            pip install langgraph-checkpoint-sqlite
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using SqliteSaver at: {path.resolve()}")
            return SqliteSaver.from_conn_string(str(path))
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed. "
                "Run: pip install langgraph-checkpoint-sqlite\n"
                "Falling back to MemorySaver."
            )
            return MemorySaver()

    @staticmethod
    def postgres(connection_string: str = None):
        """
        PostgreSQL-backed checkpointer for distributed, concurrent deployments.
        Use for: multi-process production, cloud deployments.

        Requires: langgraph-checkpoint-postgres
            pip install langgraph-checkpoint-postgres psycopg
        """
        conn_str = connection_string or os.getenv("POSTGRES_CONNECTION_STRING")
        if not conn_str:
            raise ValueError(
                "Postgres connection string required. Set POSTGRES_CONNECTION_STRING "
                "in .env or pass it directly."
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            logger.info("Using PostgresSaver")
            return PostgresSaver.from_conn_string(conn_str)
        except ImportError:
            raise ImportError(
                "langgraph-checkpoint-postgres not installed.\n"
                "Run: pip install langgraph-checkpoint-postgres psycopg"
            )

    @staticmethod
    def auto(db_path: str = "./checkpoints.db") -> Any:
        """
        Auto-select checkpointer based on environment:
        - TEST env var set → MemorySaver
        - POSTGRES_CONNECTION_STRING set → PostgresSaver
        - Otherwise → SqliteSaver
        """
        if os.getenv("TESTING") == "true":
            return CheckpointerFactory.memory()
        if os.getenv("POSTGRES_CONNECTION_STRING"):
            return CheckpointerFactory.postgres()
        return CheckpointerFactory.sqlite(db_path)


# ── History explorer ───────────────────────────────────────────────────────

class CheckpointExplorer:
    """
    Utilities for inspecting checkpoint history on a compiled graph.
    """

    def __init__(self, app, config: dict):
        self.app = app
        self.config = config

    def list_checkpoints(self) -> list[dict]:
        """
        Return a list of all checkpoints for this thread, newest first.
        Each entry: {checkpoint_id, step, values, next_nodes}
        """
        checkpoints = []
        for state in self.app.get_state_history(self.config):
            checkpoints.append({
                "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                "step": state.metadata.get("step", -1),
                "next_nodes": list(state.next),
                "keys": list(state.values.keys()),
            })
        return checkpoints

    def print_history(self) -> None:
        """Print a human-readable summary of checkpoint history."""
        print(f"\n{'='*55}")
        print(f"Checkpoint history for thread: "
              f"{self.config['configurable']['thread_id']}")
        print(f"{'='*55}")
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            print("  No checkpoints found.")
            return
        for i, cp in enumerate(checkpoints):
            print(f"  [{i}] step={cp['step']:>3} | "
                  f"next={cp['next_nodes']} | "
                  f"id={str(cp['checkpoint_id'])[:16]}...")
        print(f"{'='*55}")

    def resume_from(self, checkpoint_id: str) -> Any:
        """
        Resume graph execution from a specific checkpoint (time-travel).

        Args:
            checkpoint_id: the checkpoint_id from list_checkpoints()

        Returns:
            Result of graph.invoke() from that checkpoint
        """
        resume_config = {
            **self.config,
            "configurable": {
                **self.config["configurable"],
                "checkpoint_id": checkpoint_id,
            }
        }
        logger.info(f"Resuming from checkpoint: {checkpoint_id}")
        return self.app.invoke(None, config=resume_config)
