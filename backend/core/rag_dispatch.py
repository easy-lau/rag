"""Runner-selection policy shared by the HTTP orchestration layer.

This module owns the runner-selection policy for the active retrieval path.
It deliberately does not import any runner implementation, so the policy can
be tested without creating model clients or retrieval sessions.
"""

from __future__ import annotations

from typing import Literal

from core.query_route_compiler import RagTaskContract, rag_task_contract_gate_reason


RagRunnerVersion = Literal["v2", "direct", "reject"]


def select_rag_runner(
    *,
    task_contract: object,
    evidence_scope_filter: dict | None,
    evidence_scope_refinement_active: bool,
    is_followup: bool,
    carryover_sources: tuple[dict, ...] | list[dict],
    selected_kb_count: int | None = None,
) -> tuple[RagRunnerVersion, str]:
    """Select exactly one active runner; invalid contracts are rejected."""


    if not isinstance(task_contract, RagTaskContract):
        return "reject", "missing_or_invalid_task_contract"
    if not task_contract.dispatch_authorized:
        return "reject", "dispatch_not_authorized"
    contract_gate_reason = rag_task_contract_gate_reason(
        task_contract,
        selected_kb_count=selected_kb_count,
    )
    if contract_gate_reason is not None:
        return "reject", f"invalid_task_contract:{contract_gate_reason}"
    if not task_contract.need_retrieval:
        if (
            task_contract.retrieval_policy == "skip"
            and task_contract.response_mode
            in {"general_chat", "writing", "platform_help"}
        ):
            return "direct", f"verified_{task_contract.response_mode}"
        return "reject", "invalid_direct_task_contract"
    if (
        task_contract.retrieval_policy != "required"
        or task_contract.response_mode not in {"grounded_qa", "writing"}
    ):
        return "reject", "invalid_retrieval_task_contract"
    if evidence_scope_filter is not None:
        return "v2", "eligible_evidence_scope_selection"
    if evidence_scope_refinement_active:
        return "v2", "eligible_evidence_scope_refinement"
    if is_followup or carryover_sources:
        return "v2", "eligible_grounded_followup"
    if task_contract.response_mode == "writing":
        return "v2", "eligible_knowledge_writing"
    return "v2", "eligible_grounded_qa"
