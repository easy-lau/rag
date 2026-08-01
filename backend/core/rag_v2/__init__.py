"""Typed building blocks for the deployment-selectable RAG v2 pipeline.

The implementation remains isolated from v1, while ``api.chat`` can select it
per deployment through ``RAG_PIPELINE_VERSION``.  Keeping the boundary explicit
allows a verified rollout to switch to v2 and a restart-only rollback to v1.
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
