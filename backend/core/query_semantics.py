"""One immutable semantic contract for a single RAG turn.

The application receives three different kinds of text during a conversation:
the current user utterance, bounded historical user utterances, and a
retrieval rendering.  They must not be treated as interchangeable.  In
particular, concatenating ``current + previous`` and parsing that sentence
again turns a history *reference* into an untraceable new fact.

``ResolvedTurnSemantics`` is the narrow, source-anchored hand-off between
understanding and planning.  It is built from an already validated
``query_analysis.v2`` graph, contains only literals that occur in the current
turn or selected historical user turns, and has no authority to choose KBs,
documents, permissions, evidence coverage, bridge edge modes, or facts.

The contract deliberately owns canonical retrieval rendering as well.  That
rendering is an output of the semantic contract; it is never fed back into a
planner or a model as a new source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core.query_analysis_contract import (
    QueryAnalysis,
    QueryAnalysisAnswerCandidate,
    QueryAnalysisBridgeCandidate,
    QueryAnalysisSourceRef,
)
from core.query_surface_structure import parse_query_surface_frame


RESOLVED_TURN_SEMANTICS_SCHEMA_VERSION = "resolved_turn_semantics.v1"
ROUTE_CLARIFICATION_CONTINUATION_SCHEMA_VERSION = "route_clarification_continuation.v1"

RequestKind = Literal[
    "single_fact",
    "finite_enumeration",
    "ordered_procedure",
    "policy_overview",
    "configuration_state",
    "configuration_procedure",
    "comparison",
    "relationship",
]

KnowledgeResource = Literal[
    "document_content",
    "document_catalog",
    "document_result",
]
KnowledgeOperation = Literal[
    "answer",
    "count",
    "list",
    "group",
    "read",
    "summarize",
    "compare",
]
KnowledgeGroupBy = Literal["none", "knowledge_base", "status", "file_type"]
KnowledgeStatusFilter = Literal[
    "any",
    "ready",
    "processing",
    "failed",
    "inactive",
    "not_ready",
]
KnowledgeAnswerForm = Literal[
    "fact",
    "enumeration",
    "procedure",
    "overview",
    "comparison",
    "judgement",
]

_REQUEST_KINDS = frozenset({
    "single_fact",
    "finite_enumeration",
    "ordered_procedure",
    "policy_overview",
    "configuration_state",
    "configuration_procedure",
    "comparison",
    "relationship",
})

_KNOWLEDGE_RESOURCES = frozenset({
    "document_content",
    "document_catalog",
    "document_result",
})
_KNOWLEDGE_OPERATIONS = frozenset({
    "answer",
    "count",
    "list",
    "group",
    "read",
    "summarize",
    "compare",
})
_KNOWLEDGE_GROUP_BY = frozenset({"none", "knowledge_base", "status", "file_type"})
_KNOWLEDGE_STATUS_FILTERS = frozenset({
    "any",
    "ready",
    "processing",
    "failed",
    "inactive",
    "not_ready",
})
_KNOWLEDGE_ANSWER_FORMS = frozenset({
    "fact",
    "enumeration",
    "procedure",
    "overview",
    "comparison",
    "judgement",
})

# This grammar identifies a *resource operation surface*, not a business
# topic.  It is used only to prevent speculative vector prefetch before V3 has
# selected an execution capability.  The model still supplies the bounded
# semantic request and the backend still validates/executes it under RBAC.
_DOCUMENT_CATALOG_RESOURCE_RE = re.compile(
    r"(?:知识库|文档|文章|资料|文件)(?:名称|名字|标题|状态|类型|数量|数目)?",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_COUNT_RE = re.compile(
    r"(?:多少|几(?:个|篇|份|条)?|数量|数目|总数|合计)",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_LIST_RE = re.compile(
    r"(?:哪些|哪几|名称|名字|标题|列表|清单|列出|展示)",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_GROUP_RE = re.compile(
    r"(?:每个|各个|分别|按.+(?:分组|统计))",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_STATUS_RE = re.compile(
    r"(?:状态|处理中|处理完成|尚未完成|未完成|失败|停用|禁用)",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_FILE_TYPE_RE = re.compile(
    r"(?:文件类型|文档类型|格式|扩展名)",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_KB_GROUP_RE = re.compile(
    r"(?:每个|各个|按)\s*(?:知识库|库)",
    re.IGNORECASE,
)
_DOCUMENT_CATALOG_SUBJECT_RE = re.compile(
    r"(?:关于|有关|针对|围绕)\s*"
    r"(?P<subject>[^，,。；;！？?]{1,96}?)\s*"
    r"(?:相关)?的\s*(?:失败|处理中|处理完成|未完成|停用|禁用)?\s*"
    r"(?:知识库|文档|文章|资料|文件)",
    re.IGNORECASE,
)

# These patterns describe question form, not any company-specific fact.  The
# distinction is intentionally made here once, rather than by separate
# "contains 配置" rules in the local planner, conversation resolver and route
# compiler.
_CONFIGURATION_ACTION_RE = re.compile(
    r"(?:配置|设置|参数|开关|启用|停用|开启|关闭)",
    re.IGNORECASE,
)
_CONFIGURATION_PROCEDURE_RE = re.compile(
    # Keep the procedure-vs-state distinction independent from any product
    # vocabulary.  Both natural Chinese orderings are common:
    # ``怎么修改参数`` and ``修改参数应该怎么办``.  The latter was
    # previously classified as an ordered procedure even when the source only
    # contained a key/value configuration assignment, making rerank failure
    # erase otherwise usable deterministic evidence.
    r"(?:"
    r"(?:如何|怎么|怎样)\s*(?:进行|完成|实现|操作)?\s*"
    r"(?:配置|设置|修改|更改|调整|变更|启用|停用|开启|关闭|改)"
    r"|"
    r"(?:配置|设置|修改|更改|调整|变更|启用|停用|开启|关闭|改)"
    r"[^。！？?]{0,32}(?:怎么办|怎么做|如何处理|怎样处理)"
    r")",
    re.IGNORECASE,
)
_CONFIGURATION_STATE_RE = re.compile(
    r"(?:配置|设置|参数|开关)\s*(?:什么|哪些|多少|为?何|是?什么|有哪些)?$|"
    r"(?:需要|要|应当|应该)\s*(?:配置|设置)\s*(?:什么|哪些)?$",
    re.IGNORECASE,
)
_POLICY_OVERVIEW_RE = re.compile(
    r"(?:制度|政策|规范|规定|办法|细则|规则|标准|管理要求)\s*"
    r"(?:是什么|有哪些|包括什么|包含什么|主要内容|完整内容)?$|"
    r"(?:全部|所有|完整|总体|整体|主要内容|概述|概览)",
    re.IGNORECASE,
)


def _normalised(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique(values: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalised(raw)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return tuple(output)


@dataclass(frozen=True)
class RouteScopePartition:
    """One server-resolved applicability partition for a clarification reply.

    This is deliberately separate from both the user's answer target and the
    retrieval rendering.  A selected product/version/project may constrain an
    answer task, but it can never become a sibling answer requirement merely
    because several choices are rendered with punctuation.
    """

    product: str | None = None
    version: str | None = None
    project: str | None = None

    def __post_init__(self) -> None:
        values = {
            field: (_normalised(getattr(self, field)) or None)
            for field in ("product", "version", "project")
        }
        if not any(values.values()):
            raise ValueError("route scope partition requires an applicability value")
        for field, value in values.items():
            if value is not None and len(value) > 500:
                raise ValueError("route scope partition value is too long")
            object.__setattr__(self, field, value)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "product": self.product,
            "version": self.version,
            "project": self.project,
        }


@dataclass(frozen=True)
class RouteClarificationContinuation:
    """Structured hand-off for a reply that fills a pending route slot.

    ``current_answer`` is the literal latest user input and
    ``original_query`` is the immutable task root.  The combined rendering is
    intentionally produced once for retrieval/generation only; it must never
    be parsed as a new user sentence by V3 or the local query planner.
    """

    schema_version: Literal["route_clarification_continuation.v1"]
    original_query: str
    current_answer: str
    prior_answers: tuple[str, ...] = ()
    canonical_retrieval_query: str = ""
    scope_partitions: tuple[RouteScopePartition, ...] = ()

    def __post_init__(self) -> None:
        original = _normalised(self.original_query)
        current = _normalised(self.current_answer)
        answers = tuple(_normalised(value) for value in self.prior_answers)
        if self.schema_version != ROUTE_CLARIFICATION_CONTINUATION_SCHEMA_VERSION:
            raise ValueError("unsupported clarification continuation schema")
        if not original or not current:
            raise ValueError("clarification continuation requires task and answer")
        if any(not value for value in answers):
            raise ValueError("clarification continuation answers cannot be empty")
        partitions = tuple(self.scope_partitions)
        if len(partitions) > 8 or any(
            not isinstance(item, RouteScopePartition) for item in partitions
        ):
            raise ValueError("clarification continuation scope partitions are invalid")
        partition_keys = tuple(
            (item.product, item.version, item.project) for item in partitions
        )
        if len(set(partition_keys)) != len(partition_keys):
            raise ValueError("clarification continuation scope partitions must be unique")
        canonical = _normalised(self.canonical_retrieval_query)
        if not canonical:
            qualifiers = "、".join(_unique([*answers, current]))
            canonical = f"{original}（已确认补充条件：{qualifiers}）"
        object.__setattr__(self, "original_query", original)
        object.__setattr__(self, "current_answer", current)
        object.__setattr__(self, "prior_answers", answers[-6:])
        object.__setattr__(self, "canonical_retrieval_query", canonical[:16000])
        object.__setattr__(self, "scope_partitions", partitions)

    @property
    def semantic_query(self) -> str:
        """Return the task root used by semantic analysis/planning."""

        return self.original_query

    @property
    def answers(self) -> tuple[str, ...]:
        return (*self.prior_answers, self.current_answer)[-6:]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_query": self.original_query,
            "current_answer": self.current_answer,
            "prior_answers": list(self.prior_answers),
            "canonical_retrieval_query": self.canonical_retrieval_query,
            "scope_partitions": [item.to_dict() for item in self.scope_partitions],
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_query_length": len(self.original_query),
            "answer_count": len(self.answers),
            "scope_partition_count": len(self.scope_partitions),
            "canonical_query_length": len(self.canonical_retrieval_query),
        }

    # Compatibility for internal callers written against the former tuple
    # return value.  New code should use named fields; yielding the same three
    # projections keeps rolling upgrades harmless without restoring text
    # concatenation as a semantic input.
    def __iter__(self):
        yield self.canonical_retrieval_query
        yield self.original_query
        yield self.answers


def _request_kind(
    *,
    question: str,
    answer_count: int,
) -> RequestKind:
    """Classify only the answer contract implied by current wording.

    A model may split answer targets, but it may not label a request as a
    policy, procedure, or configuration operation.  This is a small trusted
    grammar so downstream coverage rules stay deterministic.
    """

    text = _normalised(question)
    frame = parse_query_surface_frame(text)
    operator = frame.question_operator if frame is not None else "unknown"
    if operator == "comparison":
        return "comparison"
    if operator == "relation":
        return "relationship"
    if _CONFIGURATION_PROCEDURE_RE.search(text):
        return "configuration_procedure"
    if _CONFIGURATION_ACTION_RE.search(text) and _CONFIGURATION_STATE_RE.search(
        text.rstrip("？?。.!！")
    ):
        return "configuration_state"
    if operator == "process":
        return "ordered_procedure"
    if answer_count > 1 or operator == "enumeration":
        return "finite_enumeration"
    # A governing-policy head is broad only when the current question does
    # not already bind it to a concrete entity/population/condition.  Thus
    # ``公司出差管理标准是什么`` may require document-policy closure, while
    # ``普通员工的出差标准是什么`` remains one scoped fact (possibly with an
    # optional classification augmentation).
    if _POLICY_OVERVIEW_RE.search(text) and not (
        frame is not None and frame.qualifiers
    ):
        return "policy_overview"
    return "single_fact"


def answer_shape_for_request_kind(kind: RequestKind) -> str:
    """Map semantic request kind to the existing V2 planning shape.

    This is intentionally one-way.  Existing V2 execution/evidence code can
    continue consuming ``QueryPlanV2.answer_shape`` while semantic ownership
    remains centralized here.
    """

    if kind == "finite_enumeration":
        return "multi_part"
    if kind in {"ordered_procedure", "configuration_procedure"}:
        return "process"
    if kind == "policy_overview":
        return "overview"
    if kind == "comparison":
        return "comparison"
    if kind == "relationship":
        return "fact"
    return "fact"


def request_kind_for_question(question: object, *, answer_count: int = 1) -> RequestKind:
    """Return the shared, business-agnostic request kind for one question.

    V3 and the local planner must make the same distinction between an
    ordered human procedure and a configuration assignment.  Keeping this
    small public projection here prevents one compiler from treating every
    ``如何修改`` question as an ordered-step collection while another stage
    already knows it is a source-authored key/value lookup.
    """

    count = int(answer_count)
    if count < 1:
        count = 1
    return _request_kind(
        question=_normalised(question),
        answer_count=count,
    )


def reconcile_answer_form(
    question: object,
    requested: object,
    *,
    candidate_count: int = 1,
) -> KnowledgeAnswerForm:
    """Apply the shared surface floor to a model-selected answer form.

    The model may choose the semantic form, but a plainly expressed process,
    comparison, enumeration or judgement operator must not be downgraded to a
    scalar fact by a provider that omitted or misunderstood one field.  This
    is a domain-neutral structural invariant, not a topic-specific rule.
    """

    value = str(requested or "fact").strip().casefold()
    if value not in _KNOWLEDGE_ANSWER_FORMS:
        value = "fact"
    frame = parse_query_surface_frame(_normalised(question))
    operator = frame.question_operator if frame is not None else "unknown"
    if operator == "process":
        return "procedure"
    if operator == "comparison":
        return "comparison"
    if operator == "judgement":
        return "judgement"
    if operator == "enumeration" and candidate_count == 1:
        return "enumeration"
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class KnowledgeRequestSemantics:
    """Source-bound knowledge capability requested for one turn.

    The model may choose only these closed enums and catalog-issued filter
    spans.  It cannot provide SQL, KB/document ids, permission scope or facts.
    ``filter_terms`` are exact user literals resolved by the server and are
    carried only so a capability executor never has to parse presentation text.
    """

    resource: KnowledgeResource
    operation: KnowledgeOperation
    filter_span_ids: tuple[str, ...] = ()
    filter_terms: tuple[str, ...] = ()
    group_by: KnowledgeGroupBy = "none"
    status_filter: KnowledgeStatusFilter = "any"
    result_handles: tuple[str, ...] = ()
    result_labels: tuple[str, ...] = ()
    # The answer form is semantic intent, not execution authority.  V3 may
    # choose only this closed enum; the backend compiles it into the existing
    # bounded answer-shape and evidence-coverage contracts.  It contains no
    # topic keyword, fact, scope, document id or retrieval instruction.
    answer_form: KnowledgeAnswerForm = "fact"

    def __post_init__(self) -> None:
        if self.resource not in _KNOWLEDGE_RESOURCES:
            raise ValueError("unsupported knowledge resource")
        if self.operation not in _KNOWLEDGE_OPERATIONS:
            raise ValueError("unsupported knowledge operation")
        if self.group_by not in _KNOWLEDGE_GROUP_BY:
            raise ValueError("unsupported knowledge group_by")
        if self.status_filter not in _KNOWLEDGE_STATUS_FILTERS:
            raise ValueError("unsupported knowledge status filter")
        if self.answer_form not in _KNOWLEDGE_ANSWER_FORMS:
            raise ValueError("unsupported knowledge answer form")
        span_ids = tuple(str(value or "").strip() for value in self.filter_span_ids)
        terms = tuple(_normalised(value) for value in self.filter_terms)
        result_handles = tuple(
            str(value or "").strip() for value in self.result_handles
        )
        result_labels = tuple(
            _normalised(value) for value in self.result_labels
        )
        if (
            any(not value for value in span_ids)
            or any(not value for value in terms)
            or len(span_ids) != len(set(span_ids))
            or len(span_ids) != len(terms)
            or len(span_ids) > 8
        ):
            raise ValueError("knowledge filter provenance is invalid")
        if self.resource == "document_content":
            if (
                self.operation != "answer"
                or self.group_by != "none"
                or self.status_filter != "any"
                or span_ids
                or result_handles
            ):
                raise ValueError("document content capability shape is invalid")
        elif self.resource == "document_result":
            if (
                self.operation not in {"read", "summarize", "compare"}
                or self.group_by != "none"
                or self.status_filter != "any"
                or span_ids
                or not result_handles
                or len(result_handles) != len(set(result_handles))
                or len(result_handles) != len(result_labels)
                or any(not value for value in result_handles)
                or any(not value for value in result_labels)
                or len(result_handles) > 4
                or (self.operation in {"read", "summarize"} and len(result_handles) != 1)
                or (self.operation == "compare" and len(result_handles) < 2)
            ):
                raise ValueError("document result capability shape is invalid")
        elif self.operation not in {"count", "list", "group"}:
            raise ValueError("document catalog capability shape is invalid")
        elif self.operation == "group" and self.group_by == "none":
            raise ValueError("group operation requires group_by")
        elif self.operation != "group" and self.group_by != "none":
            raise ValueError("only group operation may set group_by")
        if self.resource == "document_catalog" and result_handles:
            raise ValueError("document catalog cannot select result handles")
        object.__setattr__(self, "filter_span_ids", span_ids)
        object.__setattr__(self, "filter_terms", terms)
        object.__setattr__(self, "result_handles", result_handles)
        object.__setattr__(self, "result_labels", result_labels)

    @property
    def is_catalog_operation(self) -> bool:
        return self.resource == "document_catalog"

    @property
    def is_result_operation(self) -> bool:
        return self.resource == "document_result"

    def to_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "operation": self.operation,
            "filter_span_ids": list(self.filter_span_ids),
            "filter_terms": list(self.filter_terms),
            "group_by": self.group_by,
            "status_filter": self.status_filter,
            "result_handles": list(self.result_handles),
            "result_labels": list(self.result_labels),
            "answer_form": self.answer_form,
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "operation": self.operation,
            "filter_count": len(self.filter_terms),
            "group_by": self.group_by,
            "status_filter": self.status_filter,
            "result_count": len(self.result_handles),
            "answer_form": self.answer_form,
        }


def content_knowledge_request() -> KnowledgeRequestSemantics:
    return KnowledgeRequestSemantics(
        resource="document_content",
        operation="answer",
    )


def document_catalog_surface_operation(
    question: object,
) -> Literal["count", "list", "group"] | None:
    """Return an explicit catalog surface without choosing execution facts.

    This is a conservative preflight used to suppress vector prefetch.  A
    request executes the catalog capability only after the strict V3 contract
    selects it; an uncertain surface returns ``None`` and grants nothing.
    """

    text = _normalised(question)
    if not text or not _DOCUMENT_CATALOG_RESOURCE_RE.search(text):
        return None
    if _DOCUMENT_CATALOG_GROUP_RE.search(text):
        return "group"
    if _DOCUMENT_CATALOG_COUNT_RE.search(text):
        return "count"
    if _DOCUMENT_CATALOG_LIST_RE.search(text) or _DOCUMENT_CATALOG_STATUS_RE.search(text):
        return "list"
    return None


def document_catalog_request_for_question(
    question: object,
) -> KnowledgeRequestSemantics | None:
    """Build the conservative catalog capability floor for a clear surface.

    This is used only when strict semantic compilation is unavailable.  It
    derives a closed capability from the shared resource grammar; it never
    authorises scope, filters, documents, or content retrieval.
    """

    operation = document_catalog_surface_operation(question)
    if operation is None:
        return None
    text = _normalised(question)
    status_filter: KnowledgeStatusFilter = "any"
    status_values = (
        ("not_ready", r"(?:未完成|尚未完成|未就绪|未准备好)"),
        ("processing", r"(?:处理中|处理中的?)"),
        ("failed", r"(?:失败|处理失败)"),
        ("inactive", r"(?:停用|禁用|已停用)"),
        ("ready", r"(?:已完成|处理完成|已就绪|就绪)"),
    )
    for candidate, pattern in status_values:
        if re.search(pattern, text, flags=re.IGNORECASE):
            status_filter = candidate  # type: ignore[assignment]
            break
    group_by: KnowledgeGroupBy = "none"
    if operation == "group":
        text = _normalised(question)
        if _DOCUMENT_CATALOG_KB_GROUP_RE.search(text):
            group_by = "knowledge_base"
        elif _DOCUMENT_CATALOG_STATUS_RE.search(text):
            group_by = "status"
        elif _DOCUMENT_CATALOG_FILE_TYPE_RE.search(text):
            group_by = "file_type"
        else:
            return None
    subject_match = _DOCUMENT_CATALOG_SUBJECT_RE.search(text)
    filter_terms = (
        (subject_match.group("subject").strip(),)
        if subject_match is not None and subject_match.group("subject").strip()
        else ()
    )
    return KnowledgeRequestSemantics(
        resource="document_catalog",
        operation=operation,
        filter_span_ids=("floor_1",) if filter_terms else (),
        filter_terms=filter_terms,
        group_by=group_by,
        status_filter=status_filter,
        answer_form=("enumeration" if operation == "list" else "fact"),
    )


@dataclass(frozen=True)
class ResolvedAnswerUnit:
    """One literal answer target with its source-anchored qualifiers."""

    id: str
    target_source_ref: QueryAnalysisSourceRef
    qualifier_source_refs: tuple[QueryAnalysisSourceRef, ...]
    bridge_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("resolved answer unit requires an id")
        if self.target_source_ref.turn_key != "current":
            raise ValueError("resolved answer target must come from current turn")
        if any(
            not isinstance(value, QueryAnalysisSourceRef)
            for value in self.qualifier_source_refs
        ):
            raise ValueError("resolved answer qualifiers must be source refs")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "target_source_ref": self.target_source_ref.to_dict(),
            "qualifier_source_refs": [
                value.to_dict() for value in self.qualifier_source_refs
            ],
            "bridge_candidate_ids": list(self.bridge_candidate_ids),
        }

    def canonical_retrieval_query(self) -> str:
        """Render a retrieval query from source literals only.

        This method intentionally does not read assistant text, issue a model
        call, infer aliases, or resolve a bridge value.  It therefore cannot
        manufacture a semantic constraint that was absent from the original
        user turns.
        """

        return " ".join(_unique([
            *(item.span for item in self.qualifier_source_refs),
            self.target_source_ref.span,
        ]))


@dataclass(frozen=True)
class ResolvedTurnSemantics:
    """Immutable semantic truth for a request before retrieval planning."""

    schema_version: Literal["resolved_turn_semantics.v1"]
    relation: Literal["new", "followup", "correction", "continuation"]
    self_contained: bool
    selected_context_turn_keys: tuple[str, ...]
    request_kind: RequestKind
    answer_units: tuple[ResolvedAnswerUnit, ...]
    bridge_candidates: tuple[QueryAnalysisBridgeCandidate, ...]
    canonical_retrieval_queries: tuple[str, ...]
    canonical_retrieval_query: str
    confidence: float = 0.0
    knowledge_request: KnowledgeRequestSemantics | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RESOLVED_TURN_SEMANTICS_SCHEMA_VERSION:
            raise ValueError("unsupported resolved turn semantics schema")
        if self.relation not in {"new", "followup", "correction", "continuation"}:
            raise ValueError("unsupported resolved turn relation")
        if self.request_kind not in _REQUEST_KINDS:
            raise ValueError("unsupported resolved request kind")
        if not self.answer_units:
            raise ValueError("resolved semantics requires answer units")
        if len({item.id for item in self.answer_units}) != len(self.answer_units):
            raise ValueError("resolved semantics answer unit ids must be unique")
        selected = tuple(self.selected_context_turn_keys)
        if len(selected) != len(set(selected)):
            raise ValueError("resolved semantics context keys must be unique")
        history_keys = {
            source.turn_key
            for unit in self.answer_units
            for source in unit.qualifier_source_refs
            if source.turn_key != "current"
        }
        if self.self_contained:
            if selected or history_keys:
                raise ValueError("self-contained semantics cannot bind history")
        else:
            if self.relation == "new" or not selected or set(selected) != history_keys:
                raise ValueError("contextual semantics must exactly bind history refs")
        queries = _unique(list(self.canonical_retrieval_queries))
        if not queries:
            raise ValueError("resolved semantics requires canonical retrieval queries")
        canonical = _normalised(self.canonical_retrieval_query)
        if not canonical:
            raise ValueError("resolved semantics requires a canonical retrieval query")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= float(self.confidence) <= 1
        ):
            raise ValueError("resolved semantics confidence must be between 0 and 1")
        knowledge_request = self.knowledge_request or content_knowledge_request()
        if not isinstance(knowledge_request, KnowledgeRequestSemantics):
            raise ValueError("resolved semantics requires a knowledge request")
        object.__setattr__(self, "selected_context_turn_keys", selected)
        object.__setattr__(self, "canonical_retrieval_queries", queries)
        object.__setattr__(self, "canonical_retrieval_query", canonical)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "knowledge_request", knowledge_request)

    @property
    def answer_shape(self) -> str:
        return answer_shape_for_request_kind(self.request_kind)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "selected_context_turn_keys": list(self.selected_context_turn_keys),
            "request_kind": self.request_kind,
            "answer_units": [unit.to_dict() for unit in self.answer_units],
            "bridge_candidates": [item.to_dict() for item in self.bridge_candidates],
            "canonical_retrieval_queries": list(self.canonical_retrieval_queries),
            "canonical_retrieval_query": self.canonical_retrieval_query,
            "confidence": self.confidence,
            "knowledge_request": self.knowledge_request.to_dict(),
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "context_turn_count": len(self.selected_context_turn_keys),
            "request_kind": self.request_kind,
            "answer_unit_count": len(self.answer_units),
            "bridge_candidate_count": len(self.bridge_candidates),
            "canonical_query_count": len(self.canonical_retrieval_queries),
            "confidence": self.confidence,
            "knowledge_request": self.knowledge_request.safe_summary(),
        }


def resolve_turn_semantics(
    analysis: QueryAnalysis,
    *,
    current_question: str,
) -> ResolvedTurnSemantics:
    """Compile a validated analysis graph into the single semantic hand-off.

    ``QueryAnalysis`` has already verified exact source offsets and historical
    binding.  This function adds only deterministic request shape and a
    canonical retrieval rendering; it never expands the model's source set.
    """

    if not isinstance(analysis, QueryAnalysis):
        raise ValueError("analysis must be a QueryAnalysis")
    question = str(current_question or "")
    if not question.strip():
        raise ValueError("current_question must not be empty")
    units = tuple(
        ResolvedAnswerUnit(
            id=item.id,
            target_source_ref=item.target_source_ref,
            qualifier_source_refs=item.qualifier_source_refs,
            bridge_candidate_ids=item.bridge_candidate_ids,
        )
        for item in analysis.answer_candidates
    )
    queries = _unique([unit.canonical_retrieval_query() for unit in units])
    # A multi-target request is executed as independently closed answer tasks;
    # joining their source-only renderings is only a first-pass recall seed.
    # Task-level queries remain separate in the compiled DAG.
    canonical = "；".join(queries)
    return ResolvedTurnSemantics(
        schema_version=RESOLVED_TURN_SEMANTICS_SCHEMA_VERSION,
        relation=analysis.relation,
        self_contained=analysis.self_contained,
        selected_context_turn_keys=analysis.context_turn_keys,
        request_kind=_request_kind(question=question, answer_count=len(units)),
        answer_units=units,
        bridge_candidates=analysis.bridge_candidates,
        canonical_retrieval_queries=queries,
        canonical_retrieval_query=canonical,
        confidence=analysis.confidence,
    )


__all__ = [
    "RESOLVED_TURN_SEMANTICS_SCHEMA_VERSION",
    "ROUTE_CLARIFICATION_CONTINUATION_SCHEMA_VERSION",
    "RequestKind",
    "KnowledgeGroupBy",
    "KnowledgeAnswerForm",
    "KnowledgeOperation",
    "KnowledgeRequestSemantics",
    "KnowledgeResource",
    "KnowledgeStatusFilter",
    "ResolvedAnswerUnit",
    "ResolvedTurnSemantics",
    "RouteClarificationContinuation",
    "RouteScopePartition",
    "answer_shape_for_request_kind",
    "request_kind_for_question",
    "reconcile_answer_form",
    "content_knowledge_request",
    "document_catalog_surface_operation",
    "document_catalog_request_for_question",
    "resolve_turn_semantics",
]
