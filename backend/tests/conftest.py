"""Deterministic test-only runtime switches.

The production default enables V3 structured understanding.  Most legacy unit
fixtures intentionally exercise the deterministic/legacy hand-off with no
model credentials, so they must opt out globally rather than accidentally
making a network call.  V3 integration tests enable the mode explicitly on
their patched settings object.
"""

from __future__ import annotations

import os


# Old unit fixtures exercise the legacy context/route projection and often
# deliberately omit a V3 model fixture.  Disable the *entire* V3 semantic
# entry for that default test baseline; V3-specific tests opt in explicitly
# with a patched settings object.  Leaving ``rag_semantic_entry=v3`` while
# only turning the model mode off creates a hybrid that production never uses
# and makes legacy tests silently lose their historical projection.
os.environ["RAG_SEMANTIC_ENTRY"] = "legacy"
os.environ["RAG_QUERY_UNDERSTANDING_V3_MODE"] = "off"
