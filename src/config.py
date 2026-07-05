"""
config.py — Centralised configuration loader.

All agents import their settings from here. Values are read from
environment variables (populated via a .env file using python-dotenv).
Sensible defaults are provided so the project works out of the box.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")


# ── Ollama Model Settings ──────────────────────────────────────────────────

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.8"))
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))


# ── RAG Agent Settings ─────────────────────────────────────────────────────

PDF_PATH: Path = Path(os.getenv("PDF_PATH", ""))
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(_root / "db"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "document_content")
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))


# ── Model Validation ──────────────────────────────────────────────────────
# Set to false in test environments where Ollama may not be running
OLLAMA_VALIDATE_ON_INIT: bool = os.getenv("OLLAMA_VALIDATE_ON_INIT", "true").lower() == "true"


# ── LangSmith Tracing ──────────────────────────────────────────────────────
# LangSmith is enabled automatically when LANGCHAIN_TRACING_V2=true and
# LANGCHAIN_API_KEY are present in the environment. No code changes needed.

LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "langgraph-agent-patterns")

# Propagate tracing vars so LangChain picks them up automatically
if LANGCHAIN_TRACING_V2.lower() == "true" and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
