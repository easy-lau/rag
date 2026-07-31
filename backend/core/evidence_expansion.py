"""有界的文档内证据扩展。

该模块只扩展首轮已经定位到的文档，不自行决定答案资格。返回的候选仍必须经过
最终联合重排和证据覆盖门控，结构邻居尤其不能因为位置相近而直接进入生成上下文。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.rag_trace import content_fields, exception_log_text, trace_event
from core.retriever import (
    MAX_SCOPED_DOCUMENTS,
    MAX_SCOPED_QUERIES,
    MAX_STRUCTURAL_SEEDS,
    fetch_small_document_candidates,
    fetch_structural_neighbors,
    search_within_documents,
)


logger = logging.getLogger(__name__)


MAX_EXPANSION_ADDITIONS = 12
MAX_JOINT_CANDIDATES = 30
MAX_EXPANSION_CHARS = 16_000


@dataclass(frozen=True)
class ExpansionBudget:
    """二次检索的硬预算；调用方可以收紧，但不能突破模块上限。"""

    max_seed_documents: int = 3
    max_seed_chunks: int = 4
    max_secondary_queries: int = 2
    semantic_hits_per_document: int = 4
    max_semantic_candidates: int = 8
    neighbor_radius: int = 1
    same_section_per_seed: int = 2
    table_sibling_radius: int = 1
    max_structural_candidates: int = 8
    max_added_candidates: int = 12
    max_joint_candidates: int = 30
    max_added_chars: int = 16_000


@dataclass(frozen=True)
class CandidateMergeOutcome:
    candidates: list[dict]
    added_candidate_count: int
    added_chars: int
    deduplicated_count: int
    budget_dropped_count: int
    counts_by_origin: dict[str, int]


@dataclass(frozen=True)
class ExpansionOutcome:
    candidates: list[dict]
    seed_candidates: list[dict]
    scoped_candidates: list[dict]
    structural_candidates: list[dict]
    counts_by_origin: dict[str, int]
    added_candidate_count: int
    added_chars: int
    deduplicated_count: int
    budget_dropped_count: int
    expanded: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    full_document_candidates: list[dict] = field(default_factory=list)


def _plan_value(plan: Any, key: str, default=None):
    if plan is None:
        return default
    if isinstance(plan, dict):
        return plan.get(key, default)
    return getattr(plan, key, default)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _bounded_unique_strings(values: Any, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in _as_list(values):
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= max(0, limit):
            break
    return result


def _candidate_key(candidate: dict) -> str:
    chunk_id = candidate.get("id")
    if chunk_id:
        return f"id:{chunk_id}"
    return f"position:{candidate.get('doc_id')}:{candidate.get('chunk_index')}"


def _canonical_origin(origin: Any) -> str:
    normalized = str(origin or "").strip()
    if normalized in {"", "current_retrieval"}:
        return "global_retrieval"
    return normalized


def _merge_strings(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_list(value):
            normalized = str(item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def _merge_ints(*values: Any) -> list[int]:
    merged: set[int] = set()
    for value in values:
        for item in _as_list(value):
            try:
                merged.add(int(item))
            except (TypeError, ValueError):
                continue
    return sorted(merged)


def _normalized_candidate(candidate: dict, *, initial: bool) -> dict:
    item = dict(candidate)
    existing_origin = item.get("candidate_origin")
    canonical = _canonical_origin(existing_origin if existing_origin else None)
    if not initial and existing_origin:
        canonical = str(existing_origin)
    item["candidate_origins"] = _merge_strings(
        item.get("candidate_origins"),
        canonical,
    )
    # 保留旧 candidate_origin 兼容现有对话上下文测试；新代码统一读取复数来源。
    item["candidate_origin"] = existing_origin or canonical
    item["expansion_seed_chunk_ids"] = _merge_strings(
        item.get("expansion_seed_chunk_ids")
    )
    item["expansion_query_indexes"] = _merge_ints(
        item.get("expansion_query_indexes")
    )
    sources = item.get("expansion_sources")
    item["expansion_sources"] = [
        dict(source) for source in _as_list(sources) if isinstance(source, dict)
    ][:12]
    return item


def _source_identity(source: dict) -> tuple:
    return (
        source.get("origin"),
        source.get("seed_chunk_id"),
        source.get("query_index"),
        source.get("distance"),
    )


def _merge_candidate(existing: dict, incoming: dict) -> None:
    existing["candidate_origins"] = _merge_strings(
        existing.get("candidate_origins"),
        incoming.get("candidate_origins"),
    )
    existing["expansion_seed_chunk_ids"] = _merge_strings(
        existing.get("expansion_seed_chunk_ids"),
        incoming.get("expansion_seed_chunk_ids"),
    )
    existing["expansion_query_indexes"] = _merge_ints(
        existing.get("expansion_query_indexes"),
        incoming.get("expansion_query_indexes"),
    )
    existing["active_channels"] = _merge_strings(
        existing.get("active_channels"),
        incoming.get("active_channels"),
    )

    known_sources = {
        _source_identity(source)
        for source in existing.get("expansion_sources", [])
        if isinstance(source, dict)
    }
    for source in incoming.get("expansion_sources", []):
        if not isinstance(source, dict):
            continue
        identity = _source_identity(source)
        if identity in known_sources or len(known_sources) >= 12:
            continue
        existing.setdefault("expansion_sources", []).append(dict(source))
        known_sources.add(identity)

    # 全局 retrieval_score 保持原量纲；文档内分数单独记录，不能用结构邻近
    # 人为抬高首轮分数。其它扩展诊断字段只在缺失时补入。
    incoming_scoped_score = incoming.get("document_scoped_score")
    existing_scoped_score = existing.get("document_scoped_score")
    try:
        incoming_numeric = float(incoming_scoped_score)
    except (TypeError, ValueError):
        incoming_numeric = None
    try:
        existing_numeric = float(existing_scoped_score)
    except (TypeError, ValueError):
        existing_numeric = None
    if incoming_numeric is not None and (
        existing_numeric is None or incoming_numeric > existing_numeric
    ):
        existing["document_scoped_score"] = incoming_scoped_score

    for key in (
        "neighbor_distance",
        "structure_key",
        "expansion_depth",
        "full_document_chunk_count",
        "full_document_char_count",
    ):
        if existing.get(key) is None and incoming.get(key) is not None:
            existing[key] = incoming[key]


def _origin_counts(candidates: list[dict]) -> dict[str, int]:
    known = (
        "global_retrieval",
        "carryover_previous_turn",
        "carryover_and_current_retrieval",
        "small_document_full",
        "document_scoped",
        "adjacent",
        "same_section",
        "table_sibling",
    )
    counts = {
        origin: sum(
            origin in (candidate.get("candidate_origins") or [])
            for candidate in candidates
        )
        for origin in known
    }
    return {key: value for key, value in counts.items() if value}


def _bounded_budget(budget: ExpansionBudget) -> ExpansionBudget:
    return ExpansionBudget(
        max_seed_documents=max(
            1, min(int(budget.max_seed_documents), MAX_SCOPED_DOCUMENTS)
        ),
        max_seed_chunks=max(1, min(int(budget.max_seed_chunks), MAX_STRUCTURAL_SEEDS)),
        max_secondary_queries=max(
            1, min(int(budget.max_secondary_queries), MAX_SCOPED_QUERIES)
        ),
        semantic_hits_per_document=max(
            1, min(int(budget.semantic_hits_per_document), 4)
        ),
        max_semantic_candidates=max(
            1, min(int(budget.max_semantic_candidates), MAX_EXPANSION_ADDITIONS)
        ),
        neighbor_radius=1 if int(budget.neighbor_radius) > 0 else 0,
        same_section_per_seed=max(0, min(int(budget.same_section_per_seed), 2)),
        table_sibling_radius=1 if int(budget.table_sibling_radius) > 0 else 0,
        max_structural_candidates=max(
            1, min(int(budget.max_structural_candidates), MAX_EXPANSION_ADDITIONS)
        ),
        max_added_candidates=max(
            1, min(int(budget.max_added_candidates), MAX_EXPANSION_ADDITIONS)
        ),
        max_joint_candidates=max(
            1, min(int(budget.max_joint_candidates), MAX_JOINT_CANDIDATES)
        ),
        max_added_chars=max(1, min(int(budget.max_added_chars), MAX_EXPANSION_CHARS)),
    )


def merge_expansion_candidates(
    initial_candidates: list[dict],
    added_candidates: list[dict],
    *,
    budget: ExpansionBudget = ExpansionBudget(),
    priority_added_candidates: list[dict] | None = None,
) -> CandidateMergeOutcome:
    """按 chunk 去重并执行添加条数、联合条数和正文字符三重预算。

    ``priority_added_candidates`` 专供已经通过完整性校验的小文档全文候选使用。
    它们优先于其它首轮候选占用联合重排池，保证一篇被选中的小文档不会只留下
    半篇；与首轮重复的片段仍保留全局召回分。普通语义/结构扩展继续受默认 12
    条新增预算限制。
    """

    bounded = _bounded_budget(budget)
    priority_added_candidates = list(priority_added_candidates or [])
    if priority_added_candidates:
        initial_by_key: dict[str, dict] = {}
        initial_order: list[str] = []
        deduplicated = 0
        dropped = 0
        for candidate in initial_candidates:
            item = _normalized_candidate(candidate, initial=True)
            identity = _candidate_key(item)
            if identity in initial_by_key:
                _merge_candidate(initial_by_key[identity], item)
                deduplicated += 1
                continue
            initial_by_key[identity] = item
            initial_order.append(identity)

        merged: dict[str, dict] = {}
        order: list[str] = []
        added_count = 0
        added_chars = 0

        # 全文候选本身已经由 retriever 按“整篇”校验并限制在联合池/字符预算内。
        # 先按文档片段顺序放入，剩余位置再保留其它首轮和普通扩展候选。
        for candidate in priority_added_candidates:
            item = _normalized_candidate(candidate, initial=False)
            identity = _candidate_key(item)
            if identity in merged:
                _merge_candidate(merged[identity], item)
                deduplicated += 1
                continue

            initial_item = initial_by_key.get(identity)
            is_added = initial_item is None
            content_chars = len(str(item.get("content") or "")) if is_added else 0
            if (
                len(order) >= bounded.max_joint_candidates
                or added_chars + content_chars > bounded.max_added_chars
            ):
                dropped += 1
                continue

            if initial_item is not None:
                initial_by_key.pop(identity, None)
                _merge_candidate(initial_item, item)
                item = initial_item
                deduplicated += 1
            else:
                item["expansion_depth"] = 1
                added_count += 1
                added_chars += content_chars
            merged[identity] = item
            order.append(identity)

        for identity in initial_order:
            item = initial_by_key.get(identity)
            if item is None:
                continue
            if len(order) >= bounded.max_joint_candidates:
                dropped += 1
                continue
            merged[identity] = item
            order.append(identity)

        for candidate in added_candidates:
            item = _normalized_candidate(candidate, initial=False)
            identity = _candidate_key(item)
            if identity in merged:
                _merge_candidate(merged[identity], item)
                deduplicated += 1
                continue

            content_chars = len(str(item.get("content") or ""))
            if (
                added_count >= bounded.max_added_candidates
                or len(order) >= bounded.max_joint_candidates
                or added_chars + content_chars > bounded.max_added_chars
            ):
                dropped += 1
                continue
            item["expansion_depth"] = 1
            merged[identity] = item
            order.append(identity)
            added_count += 1
            added_chars += content_chars

        candidates = [merged[identity] for identity in order]
        return CandidateMergeOutcome(
            candidates=candidates,
            added_candidate_count=added_count,
            added_chars=added_chars,
            deduplicated_count=deduplicated,
            budget_dropped_count=dropped,
            counts_by_origin=_origin_counts(candidates),
        )

    merged: dict[str, dict] = {}
    order: list[str] = []
    deduplicated = 0
    dropped = 0

    for candidate in initial_candidates:
        item = _normalized_candidate(candidate, initial=True)
        identity = _candidate_key(item)
        if identity in merged:
            _merge_candidate(merged[identity], item)
            deduplicated += 1
            continue
        if len(order) >= bounded.max_joint_candidates:
            dropped += 1
            continue
        merged[identity] = item
        order.append(identity)

    added_count = 0
    added_chars = 0
    for candidate in added_candidates:
        item = _normalized_candidate(candidate, initial=False)
        identity = _candidate_key(item)
        if identity in merged:
            _merge_candidate(merged[identity], item)
            deduplicated += 1
            continue

        content_chars = len(str(item.get("content") or ""))
        if (
            added_count >= bounded.max_added_candidates
            or len(order) >= bounded.max_joint_candidates
            or added_chars + content_chars > bounded.max_added_chars
        ):
            dropped += 1
            continue
        item["expansion_depth"] = 1
        merged[identity] = item
        order.append(identity)
        added_count += 1
        added_chars += content_chars

    candidates = [merged[identity] for identity in order]
    return CandidateMergeOutcome(
        candidates=candidates,
        added_candidate_count=added_count,
        added_chars=added_chars,
        deduplicated_count=deduplicated,
        budget_dropped_count=dropped,
        counts_by_origin=_origin_counts(candidates),
    )


def _select_seed_candidates(
    candidates: list[dict],
    plan: Any,
    budget: ExpansionBudget,
) -> list[dict]:
    requested_chunks = set(_bounded_unique_strings(
        _plan_value(plan, "seed_chunk_ids", []),
        budget.max_seed_chunks,
    ))
    requested_docs = set(_bounded_unique_strings(
        _plan_value(plan, "seed_doc_ids", []),
        budget.max_seed_documents,
    ))
    target_indexes: list[int] = []
    for value in _as_list(_plan_value(plan, "target_candidate_indexes", [])):
        try:
            target_index = int(value)
        except (TypeError, ValueError):
            continue
        if target_index < 1 or target_index in target_indexes:
            continue
        target_indexes.append(target_index)
        if len(target_indexes) >= budget.max_seed_chunks:
            break

    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("doc_id") is not None and candidate.get("id") is not None
    ]
    if requested_chunks:
        preferred = [
            candidate for candidate in eligible
            if str(candidate.get("id")) in requested_chunks
        ]
    elif target_indexes:
        # target_candidate_indexes 指第一次重排的原始输入序号。结果已经按证据分数
        # 重排，必须优先通过 rerank_candidate_index 回映，不能直接拿当前列表位置。
        indexed = {
            int(candidate["rerank_candidate_index"]): candidate
            for candidate in eligible
            if isinstance(candidate.get("rerank_candidate_index"), int)
            and not isinstance(candidate.get("rerank_candidate_index"), bool)
        }
        if indexed:
            preferred = [indexed[index] for index in target_indexes if index in indexed]
        else:
            preferred = [
                eligible[index - 1]
                for index in target_indexes
                if index <= len(eligible)
            ]
    elif requested_docs:
        preferred = [
            candidate for candidate in eligible
            if str(candidate.get("doc_id")) in requested_docs
        ]
    else:
        preferred = eligible

    # 模型计划中的 ID 只能选择首轮候选，全部无效时回退首轮排序，绝不按计划
    # 直接访问任意数据库 doc_id。
    if not preferred:
        preferred = eligible

    selected: list[dict] = []
    selected_docs: set[str] = set()
    for candidate in preferred:
        doc_id = str(candidate.get("doc_id"))
        if doc_id not in selected_docs and len(selected_docs) >= budget.max_seed_documents:
            continue
        selected.append(dict(candidate))
        selected_docs.add(doc_id)
        if len(selected) >= budget.max_seed_chunks:
            break
    return selected


def _bridge_terms_from_seeds(seeds: list[dict]) -> list[str]:
    terms: list[str] = []
    for seed in seeds:
        for fact in _as_list(seed.get("bridge_facts")):
            if isinstance(fact, dict):
                values = (fact.get("subject"), fact.get("object"))
            else:
                values = (getattr(fact, "subject", None), getattr(fact, "object", None))
            terms.extend(value for value in values if value)
    return _bounded_unique_strings(terms, 8)


def _expansion_queries(
    question: str,
    plan: Any,
    budget: ExpansionBudget,
    seeds: list[dict],
) -> list[str]:
    secondaries = _bounded_unique_strings(
        _plan_value(
            plan,
            "secondary_queries",
            _plan_value(plan, "queries", []),
        ),
        budget.max_secondary_queries,
    )
    bridge_terms = _bounded_unique_strings(
        _plan_value(plan, "bridge_terms", _bridge_terms_from_seeds(seeds)),
        8,
    )
    normalized_question = " ".join(str(question or "").split()).strip()

    queries: list[str] = []
    if len(secondaries) >= 2:
        queries.extend(secondaries[:2])
    elif secondaries:
        if normalized_question:
            queries.append(normalized_question)
        queries.extend(secondaries)
    elif normalized_question:
        queries.append(normalized_question)
        if bridge_terms:
            queries.append(f"{normalized_question} {' '.join(bridge_terms)}")

    return _bounded_unique_strings(queries, budget.max_secondary_queries)


def _should_expand(plan: Any) -> bool:
    explicit = _plan_value(plan, "should_expand", None)
    if explicit is None:
        explicit = _plan_value(plan, "needed", None)
    if isinstance(explicit, bool):
        return explicit
    return bool(
        _plan_value(plan, "secondary_queries", None)
        or _plan_value(plan, "queries", None)
        or _plan_value(plan, "bridge_terms", None)
        or _plan_value(plan, "seed_chunk_ids", None)
    )


async def expand_evidence_candidates(
    db: AsyncSession,
    *,
    question: str,
    kb_ids: list[uuid.UUID],
    initial_candidates: list[dict],
    plan: Any,
    method: str = "hybrid",
    budget: ExpansionBudget = ExpansionBudget(),
    trace_id: str | None = None,
    surface: str = "chat",
) -> ExpansionOutcome:
    """执行一次受限扩展，并返回供最终联合重排使用的候选池。

    ``plan`` 可以是 dataclass、SimpleNamespace 或 dict，只读取公开字段，不依赖
    reranker 的具体类型。即使计划包含任意 doc_id，也只能从 ``initial_candidates``
    中选出实际种子，因而不会绕过当前知识库与首轮召回边界。
    """

    started_at = time.perf_counter()
    bounded = _bounded_budget(budget)
    seeds = _select_seed_candidates(initial_candidates, plan, bounded)
    queries = _expansion_queries(question, plan, bounded, seeds)
    seed_doc_ids = []
    for seed in seeds:
        doc_id = seed.get("doc_id")
        if doc_id is not None and str(doc_id) not in {str(value) for value in seed_doc_ids}:
            seed_doc_ids.append(doc_id)
        if len(seed_doc_ids) >= bounded.max_seed_documents:
            break
    should_expand = _should_expand(plan) and bool(seeds and queries and kb_ids)

    if trace_id:
        trace_event(
            "retrieval.expansion_planned",
            trace_id=trace_id,
            should_expand=should_expand,
            seed_document_count=len(seed_doc_ids),
            seed_chunk_count=len(seeds),
            secondary_query_count=len(queries),
            bridge_term_count=len(_bounded_unique_strings(
                _plan_value(plan, "bridge_terms", _bridge_terms_from_seeds(seeds)),
                8,
            )),
            required_facet_count=len(_as_list(_plan_value(
                plan,
                "required_facets",
                _plan_value(plan, "missing_requirement_ids", []),
            ))),
            missing_requirement_ids=_bounded_unique_strings(
                _plan_value(plan, "missing_requirement_ids", []),
                8,
            ),
            adaptive_small_document_enabled=True,
            max_full_document_candidates=bounded.max_joint_candidates,
            max_full_document_chars=bounded.max_added_chars,
            max_added_candidates=bounded.max_added_candidates,
            max_joint_rerank_candidates=bounded.max_joint_candidates,
            max_added_chars=bounded.max_added_chars,
            **content_fields(
                "expansion_queries",
                json.dumps(queries, ensure_ascii=False),
            ),
        )

    if not should_expand:
        merge = merge_expansion_candidates(
            initial_candidates,
            [],
            budget=bounded,
        )
        return ExpansionOutcome(
            candidates=merge.candidates,
            seed_candidates=seeds,
            scoped_candidates=[],
            structural_candidates=[],
            counts_by_origin=merge.counts_by_origin,
            added_candidate_count=0,
            added_chars=0,
            deduplicated_count=merge.deduplicated_count,
            budget_dropped_count=merge.budget_dropped_count,
            expanded=False,
        )

    errors: list[str] = []
    full_document: list[dict] = []
    try:
        fetched_full_document = await fetch_small_document_candidates(
            db,
            kb_ids=kb_ids,
            doc_ids=seed_doc_ids,
            max_chunks=bounded.max_joint_candidates,
            max_chars=bounded.max_added_chars,
            trace_id=trace_id,
        )
        # SQL 已强制授权范围，这里再做一层廉价防御，避免未来替换实现或测试替身
        # 把计划外文档带入联合候选池。
        allowed_doc_ids = {str(value) for value in seed_doc_ids}
        allowed_kb_ids = {str(value) for value in kb_ids}
        full_document = [
            candidate
            for candidate in fetched_full_document
            if str(candidate.get("doc_id") or "") in allowed_doc_ids
            and str(candidate.get("kb_id") or "") in allowed_kb_ids
        ]
    except Exception as exc:
        errors.append(f"small_document_full:{type(exc).__name__}")
        logger.warning(
            "[证据扩展] 小文档全文候选加载失败，继续文档内语义/结构补检 error=%s",
            exception_log_text(exc),
        )

    loaded_document_ids = {
        str(candidate.get("doc_id"))
        for candidate in full_document
        if candidate.get("doc_id") is not None
    }
    remaining_doc_ids = [
        doc_id for doc_id in seed_doc_ids
        if str(doc_id) not in loaded_document_ids
    ]
    scoped: list[dict] = []
    if remaining_doc_ids:
        try:
            scoped = await search_within_documents(
                db,
                queries=queries,
                kb_ids=kb_ids,
                doc_ids=remaining_doc_ids,
                method=method,
                per_document_limit=bounded.semantic_hits_per_document,
                total_limit=bounded.max_semantic_candidates,
                trace_id=trace_id,
                surface=surface,
            )
        except Exception as exc:
            errors.append(f"document_scoped:{type(exc).__name__}")
            logger.warning(
                "[证据扩展] 文档内检索失败，保留首轮与结构候选 error=%s",
                exception_log_text(exc),
            )

    structural_seed_pool: list[dict] = []
    seen_seed_ids: set[str] = set()
    for candidate in [*seeds, *scoped]:
        if str(candidate.get("doc_id") or "") in loaded_document_ids:
            continue
        identity = str(candidate.get("id") or "")
        if not identity or identity in seen_seed_ids:
            continue
        seen_seed_ids.add(identity)
        structural_seed_pool.append(candidate)
        if len(structural_seed_pool) >= bounded.max_seed_chunks:
            break

    structural: list[dict] = []
    if structural_seed_pool:
        try:
            structural = await fetch_structural_neighbors(
                db,
                kb_ids=kb_ids,
                seed_candidates=structural_seed_pool,
                neighbor_radius=bounded.neighbor_radius,
                same_section_limit=bounded.same_section_per_seed,
                table_sibling_radius=bounded.table_sibling_radius,
                total_limit=bounded.max_structural_candidates,
                trace_id=trace_id,
            )
        except Exception as exc:
            errors.append(f"structure:{type(exc).__name__}")
            logger.warning(
                "[证据扩展] 结构补充失败，保留首轮与文档内检索候选 error=%s",
                exception_log_text(exc),
            )

    # 文档内语义结果优先占用新增预算；结构片段是连续性补充，不能挤掉更明确
    # 的二次检索命中。最终是否可用于回答仍由联合重排决定。
    merge = merge_expansion_candidates(
        initial_candidates,
        [*scoped, *structural],
        budget=bounded,
        priority_added_candidates=full_document,
    )
    if trace_id:
        trace_event(
            "retrieval.expansion_completed",
            trace_id=trace_id,
            initial_candidate_count=len(initial_candidates),
            full_document_count=len(loaded_document_ids),
            full_document_candidate_count=len(full_document),
            semantic_fallback_document_count=len(remaining_doc_ids),
            added_candidate_count=merge.added_candidate_count,
            combined_candidate_count=len(merge.candidates),
            counts_by_origin=merge.counts_by_origin,
            deduplicated_count=merge.deduplicated_count,
            budget_dropped_count=merge.budget_dropped_count,
            added_chars=merge.added_chars,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            error_count=len(errors),
        )

    return ExpansionOutcome(
        candidates=merge.candidates,
        seed_candidates=seeds,
        scoped_candidates=scoped,
        structural_candidates=structural,
        counts_by_origin=merge.counts_by_origin,
        added_candidate_count=merge.added_candidate_count,
        added_chars=merge.added_chars,
        deduplicated_count=merge.deduplicated_count,
        budget_dropped_count=merge.budget_dropped_count,
        expanded=True,
        errors=tuple(errors),
        full_document_candidates=full_document,
    )
