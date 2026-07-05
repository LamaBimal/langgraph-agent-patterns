"""
Tests for config.py — verifies that defaults are loaded correctly
and that type coercions work as expected.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config


def test_ollama_model_default():
    """Default model should be set."""
    assert isinstance(config.OLLAMA_MODEL, str)
    assert len(config.OLLAMA_MODEL) > 0


def test_temperature_is_float():
    """Temperature must be a float."""
    assert isinstance(config.OLLAMA_TEMPERATURE, float)
    assert 0.0 <= config.OLLAMA_TEMPERATURE <= 2.0


def test_num_predict_is_int():
    """num_predict must be a positive integer."""
    assert isinstance(config.OLLAMA_NUM_PREDICT, int)
    assert config.OLLAMA_NUM_PREDICT > 0


def test_retriever_k_is_int():
    """RETRIEVER_K must be a positive integer."""
    assert isinstance(config.RETRIEVER_K, int)
    assert config.RETRIEVER_K > 0


def test_chunk_size_greater_than_overlap():
    """CHUNK_SIZE should be larger than CHUNK_OVERLAP to avoid infinite loops."""
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP


def test_chroma_collection_is_string():
    """Chroma collection name must be a non-empty string."""
    assert isinstance(config.CHROMA_COLLECTION, str)
    assert len(config.CHROMA_COLLECTION) > 0


def test_embedding_model_set():
    """Embedding model name must be a non-empty string."""
    assert isinstance(config.OLLAMA_EMBEDDING_MODEL, str)
    assert len(config.OLLAMA_EMBEDDING_MODEL) > 0
