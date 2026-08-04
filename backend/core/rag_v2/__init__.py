"""Typed building blocks for the deployment-selectable RAG v2 pipeline.

The implementation is the active evidence runner.  ``api.chat`` keeps the
module boundary for diagnostics, while requests no longer switch to a legacy
retrieval runner.
"""

from core.rag_v2.contracts import (
    ANSWER_SHAPES,
    QUERY_PLAN_V2_SCHEMA_VERSION,
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
    QueryPlanV2,
)
from core.rag_v2.query_plan import plan_query_locally

__all__ = [
    "ANSWER_SHAPES",
    "QUERY_PLAN_V2_SCHEMA_VERSION",
    "AnswerRequirementV2",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceState",
    "QueryPlanV2",
    "plan_query_locally",
]
