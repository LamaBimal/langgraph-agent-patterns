"""
conftest.py — Pytest configuration and shared fixtures.

Sets up the Python path so tests can import from src/ without installing
the package.
"""

import sys
import os

# Ensure agents/ is on the path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
