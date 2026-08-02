"""Deterministic defaults for the legacy unittest fixture suite.

The repository uses both ``pytest`` and ``python -m unittest``.  Pytest loads
``conftest.py`` automatically, while unittest does not; without this package
initializer the latter accidentally booted the production V3 entry and could
attempt model/DB work from legacy isolated fakes.  Tests that exercise V3
always provide an explicit patched settings object, so the default remains a
single legacy semantic authority for old fixtures.
"""

from __future__ import annotations

import os


os.environ["RAG_SEMANTIC_ENTRY"] = "legacy"
os.environ["RAG_QUERY_UNDERSTANDING_V3_MODE"] = "off"
