"""Explicit retrieval-task DAG contracts for RAG v2.

The legacy plan contains a flat ``retrieval_queries`` tuple.  A tuple cannot
express which query proves which requirement, nor can it express the
``bridge -> answer`` ordering needed by multi-hop questions.  This module is
the deliberately small contract that replaces that implicit positional
relationship for later execution work.

The compiler is deterministic and *plan-only*: it does not inspect the flat
query list and it does not ask a model to invent facts.  Answer queries are
copied verbatim from their requirement descriptions.  Bridge queries contain
only the source-authored bridge subject (and an optional source-authored
scope) plus a fixed, domain-neutral classification vocabulary.  Retrieval
deduplication is intentionally not performed here; two tasks with the same
query remain distinct because their evidence roles and requirement ownership
are different.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from core.rag_v2.contracts import (
    AnswerRequirementV2,
    QueryPlanV2,
    validate_answer_requirement_graph,
)
from core.query_constraints import ApplicabilityScope


RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION = "retrieval_task_graph.v2"

TaskRole = Literal["answer", "bridge", "anchor"]
_TASK_ROLES = frozenset({"answer", "bridge", "anchor"})
BridgeEdgeMode = Literal["proof", "augmentation"]
_BRIDGE_EDGE_MODES = frozenset({"proof", "augmentation"})
# A V2 execution is either fully ledgered or deliberately not runnable.  There
# is no third, "mostly compatible" mode: an execution that lacks a graph and
# request-local provenance has no way to prove an answer route.
ExecutionMode = Literal["ledgered", "not_ready"]
_EXECUTION_MODES = frozenset({"ledgered", "not_ready"})
_TASK_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_TASKS = 32
_MAX_QUERY_CHARS = 1000
_MAX_SCOPE_CHARS = 200

# These are intentionally fixed lexical baskets.  They describe only the
# relation family already proven by the planner/validator; they never assert a
# concrete grade, amount, product, or business synonym.  Keeping the baskets
# separate prevents a condition/mapping edge from being retrieved as though it
# were an employee-grade taxonomy lookup.
_BRIDGE_QUERY_SUFFIX_BY_KIND = {
    "classification": "对应的适用分类 等级 类别 职级 角色 版本 档位 阶段",
    "condition": "适用条件 决定关系 范围 期间 地区 区域 阶段",
    "mapping": "对应关系 映射关系 归属 适用关系",
}


def _text(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return normalized


def _optional_text(value: object, *, field: str, max_chars: int) -> str | None:
    if value is None:
        return None
    return _text(value, field=field, max_chars=max_chars)


def _ids(value: object, *, field: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple")
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = _text(raw, field=field, max_chars=80)
        if not pattern.fullmatch(item):
            raise ValueError(f"{field} contains an invalid identifier")
        if item in seen:
            raise ValueError(f"{field} contains duplicate identifiers")
        seen.add(item)
        output.append(item)
    return tuple(output)


def _bridge_edge_modes(value: object) -> Mapping[str, BridgeEdgeMode]:
    """Normalise answer-to-bridge edge semantics into an immutable mapping.

    ``dependency_task_ids`` carries scheduling topology, while this mapping
    records why an answer consumes a bridge node.  Keeping the two dimensions
    separate prevents a scheduler or ledger from guessing that every bridge
    parent is a proof prerequisite.
    """

    if not isinstance(value, Mapping):
        raise ValueError("bridge_edge_modes must be a mapping")
    normalized: dict[str, BridgeEdgeMode] = {}
    for raw_task_id, raw_mode in value.items():
        task_id = _text(raw_task_id, field="bridge edge task id", max_chars=80)
        if not _TASK_ID_RE.fullmatch(task_id):
            raise ValueError("bridge edge task id must be a stable task identifier")
        if not isinstance(raw_mode, str):
            raise ValueError("bridge edge mode must be a string")
        mode = raw_mode.strip().casefold()
        if mode not in _BRIDGE_EDGE_MODES:
            raise ValueError("bridge edge mode must be proof or augmentation")
        normalized[task_id] = mode  # type: ignore[assignment]
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class RetrievalTask:
    """One explicit retrieval operation in the task DAG.

    ``answer`` and ``bridge`` tasks each target exactly one requirement.  An
    ``anchor`` task is a recall-only seed and therefore must target no
    requirement and can never be marked required.  The executor may later
    physically merge equal query strings, but it must retain these task
    identities and ownership edges.
    """

    task_id: str
    role: TaskRole
    query: str
    target_requirement_ids: tuple[str, ...] = ()
    dependency_task_ids: tuple[str, ...] = ()
    # Only answer tasks may classify a bridge parent.  The graph later checks
    # that every actual bridge dependency appears exactly once in this map and
    # that no anchor/non-bridge task is mislabeled as a bridge edge.
    bridge_edge_modes: Mapping[str, BridgeEdgeMode] = field(default_factory=dict)
    # One task owns exactly one canonical applicability scope.  An anchor is
    # intentionally unscoped (or recalls a union at the executor boundary),
    # while every answer/bridge task carries the requirement-local scope used
    # for final candidate admission.
    applicability_scope: ApplicabilityScope | None = None
    # Legacy scalar projection retained for callers/tests that construct a
    # task directly.  It is normalized back into ``applicability_scope`` and
    # must never become a second source of truth.
    scope_product: str | None = None
    scope_version: str | None = None
    scope_explicit_version: bool = False
    required: bool = False

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, field="task_id", max_chars=80)
        if not _TASK_ID_RE.fullmatch(task_id):
            raise ValueError("task_id must be a stable lowercase identifier")
        if self.role not in _TASK_ROLES:
            raise ValueError("task role must be answer, bridge or anchor")
        query = _text(self.query, field="task query", max_chars=_MAX_QUERY_CHARS)
        target_requirement_ids = _ids(
            self.target_requirement_ids,
            field="target_requirement_ids",
            pattern=_REQUIREMENT_ID_RE,
        )
        dependency_task_ids = _ids(
            self.dependency_task_ids,
            field="dependency_task_ids",
            pattern=_TASK_ID_RE,
        )
        bridge_edge_modes = _bridge_edge_modes(self.bridge_edge_modes)
        if self.role == "anchor":
            if target_requirement_ids:
                raise ValueError("anchor tasks cannot target requirements")
            if self.required is not False:
                raise ValueError("anchor tasks cannot be required")
        elif len(target_requirement_ids) != 1:
            raise ValueError(
                f"{self.role} tasks must target exactly one requirement"
            )
        if self.role != "answer" and bridge_edge_modes:
            raise ValueError("only answer tasks can define bridge edge modes")
        if any(task_id not in dependency_task_ids for task_id in bridge_edge_modes):
            raise ValueError(
                "bridge edge modes must reference declared task dependencies"
            )
        if not isinstance(self.scope_explicit_version, bool):
            raise ValueError("scope_explicit_version must be a boolean")
        legacy_product = _optional_text(
            self.scope_product,
            field="scope_product",
            max_chars=_MAX_SCOPE_CHARS,
        )
        legacy_version = _optional_text(
            self.scope_version,
            field="scope_version",
            max_chars=_MAX_SCOPE_CHARS,
        )
        legacy_explicit_version = bool(self.scope_explicit_version)
        if legacy_explicit_version and legacy_version is None:
            raise ValueError(
                "scope_explicit_version requires scope_version"
            )
        supplied_scope = self.applicability_scope
        if supplied_scope is not None and not isinstance(
            supplied_scope,
            ApplicabilityScope,
        ):
            raise ValueError("applicability_scope must be an ApplicabilityScope")
        if supplied_scope is None:
            canonical_scope = ApplicabilityScope(
                product=legacy_product,
                version=legacy_version,
                explicit_version=legacy_explicit_version,
                extraction_reason="legacy_retrieval_task_scope_projection",
            )
        else:
            canonical_scope = supplied_scope
            if legacy_product is not None and legacy_product != canonical_scope.product:
                raise ValueError("scope_product conflicts with applicability_scope")
            if legacy_version is not None and legacy_version != canonical_scope.version:
                raise ValueError("scope_version conflicts with applicability_scope")
            if legacy_explicit_version and not canonical_scope.explicit_version:
                raise ValueError(
                    "scope_explicit_version conflicts with applicability_scope"
                )
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")

        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "target_requirement_ids", target_requirement_ids)
        object.__setattr__(self, "dependency_task_ids", dependency_task_ids)
        object.__setattr__(self, "bridge_edge_modes", bridge_edge_modes)
        object.__setattr__(self, "applicability_scope", canonical_scope)
        object.__setattr__(self, "scope_product", canonical_scope.product)
        object.__setattr__(self, "scope_version", canonical_scope.version)
        object.__setattr__(
            self,
            "scope_explicit_version",
            canonical_scope.explicit_version,
        )

    @property
    def scope_project(self) -> str | None:
        return self.applicability_scope.project if self.applicability_scope else None

    @property
    def scope_project_source(self):
        return (
            self.applicability_scope.project_source
            if self.applicability_scope
            else None
        )

    @property
    def scope_fingerprint(self) -> str:
        return (
            self.applicability_scope.fingerprint
            if self.applicability_scope
            else ApplicabilityScope().fingerprint
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only JSON-compatible primitive values.

        Query text is intentionally retained for explainability.  Callers
        writing privacy-sensitive traces can use :meth:`safe_summary` on the
        graph instead; this method itself never returns dataclass instances,
        sets, tuples, or executable references.
        """

        result: dict[str, Any] = {
            "task_id": self.task_id,
            "role": self.role,
            "query": self.query,
            "target_requirement_ids": list(self.target_requirement_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "required": self.required,
        }
        if self.role == "answer":
            result["bridge_edge_modes"] = {
                task_id: self.bridge_edge_modes[task_id]
                for task_id in self.dependency_task_ids
                if task_id in self.bridge_edge_modes
            }
        if (
            self.applicability_scope is not None
            and self.applicability_scope.has_scope_constraint
        ):
            result["scope"] = self.applicability_scope.as_dict()
        return result

    def bridge_parent_task_ids(
        self,
        *,
        mode: BridgeEdgeMode | None = None,
    ) -> tuple[str, ...]:
        """Return this answer's declared bridge parent ids in DAG order.

        The task graph validates that these ids actually target bridge
        requirements.  This method intentionally does not inspect natural
        language or infer a mode for a legacy dependency.
        """

        if self.role != "answer":
            return ()
        if mode is not None and mode not in _BRIDGE_EDGE_MODES:
            raise ValueError("bridge edge mode must be proof or augmentation")
        return tuple(
            dependency_id
            for dependency_id in self.dependency_task_ids
            if dependency_id in self.bridge_edge_modes
            and (mode is None or self.bridge_edge_modes[dependency_id] == mode)
        )


@dataclass(frozen=True)
class AnswerBridgePath:
    """One typed answer-to-bridge route compiled from the immutable DAG.

    The path is the sole handoff for a materialised second-hop query.  It
    carries both requirement and task identities so downstream code cannot
    reconstruct an edge from free text, a query position, or a merged
    candidate lineage.
    """

    answer_task_id: str
    answer_requirement_id: str
    bridge_task_ids: tuple[str, ...]
    bridge_requirement_ids: tuple[str, ...]
    edge_mode: BridgeEdgeMode
    bridge_edge_modes: Mapping[str, BridgeEdgeMode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        answer_task_id = _text(
            self.answer_task_id,
            field="answer bridge path answer task id",
            max_chars=80,
        )
        answer_requirement_id = _text(
            self.answer_requirement_id,
            field="answer bridge path answer requirement id",
            max_chars=80,
        )
        bridge_task_ids = _ids(
            self.bridge_task_ids,
            field="answer bridge path bridge task ids",
            pattern=_TASK_ID_RE,
        )
        bridge_requirement_ids = _ids(
            self.bridge_requirement_ids,
            field="answer bridge path bridge requirement ids",
            pattern=_REQUIREMENT_ID_RE,
        )
        edge_mode = str(self.edge_mode or "").strip().casefold()
        if edge_mode not in _BRIDGE_EDGE_MODES:
            raise ValueError("answer bridge path edge mode is invalid")
        if not bridge_task_ids or not bridge_requirement_ids:
            raise ValueError("answer bridge path requires bridge parents")
        if len(bridge_task_ids) != len(bridge_requirement_ids):
            raise ValueError(
                "answer bridge path task and requirement parent counts differ"
            )
        bridge_edge_modes = _bridge_edge_modes(self.bridge_edge_modes)
        if set(bridge_edge_modes) != set(bridge_task_ids):
            raise ValueError(
                "answer bridge path modes must cover every bridge parent"
            )
        if edge_mode == "proof" and any(
            value != "proof" for value in bridge_edge_modes.values()
        ):
            raise ValueError("proof bridge path cannot contain augmentation edges")
        if edge_mode == "augmentation" and not any(
            value == "augmentation" for value in bridge_edge_modes.values()
        ):
            raise ValueError("augmentation bridge path requires an augmentation edge")
        object.__setattr__(self, "answer_task_id", answer_task_id)
        object.__setattr__(self, "answer_requirement_id", answer_requirement_id)
        object.__setattr__(self, "bridge_task_ids", bridge_task_ids)
        object.__setattr__(self, "bridge_requirement_ids", bridge_requirement_ids)
        object.__setattr__(self, "edge_mode", edge_mode)
        object.__setattr__(self, "bridge_edge_modes", bridge_edge_modes)


def _requirement_snapshot(
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[dict[str, Any], ...]:
    """Build a content-light requirement snapshot for graph serialization."""

    snapshots: list[dict[str, Any]] = []
    for requirement in requirements:
        snapshot: dict[str, Any] = {
            "id": requirement.id,
            "role": requirement.role,
            "importance": requirement.importance,
        }
        if requirement.depends_on_requirement_ids is not None:
            snapshot["depends_on_requirement_ids"] = list(
                requirement.depends_on_requirement_ids
            )
        if requirement.augmentation_requirement_ids is not None:
            snapshot["augmentation_requirement_ids"] = list(
                requirement.augmentation_requirement_ids
            )
        if requirement.bridge_kind is not None:
            snapshot["bridge_kind"] = requirement.bridge_kind
        snapshots.append(snapshot)
    return tuple(snapshots)


@dataclass(frozen=True)
class RetrievalTaskGraph:
    """Validated retrieval DAG plus the requirement ownership it serves."""

    requirements: tuple[AnswerRequirementV2, ...]
    tasks: tuple[RetrievalTask, ...]
    schema_version: str = RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported retrieval task graph schema version")
        requirements = tuple(self.requirements)
        if any(not isinstance(item, AnswerRequirementV2) for item in requirements):
            raise ValueError("requirements must contain AnswerRequirementV2 values")
        requirement_ids = [item.id for item in requirements]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("graph contains duplicate requirement ids")
        # A graph is an executable boundary, so every answer dependency must
        # be explicit and every bridge requirement must have an answer owner.
        # This validates the typed edge data without re-parsing natural
        # language.
        validate_answer_requirement_graph(
            requirements,
            require_explicit_answer_dependencies=True,
            require_referenced_bridges=True,
        )
        if any(
            item.role == "bridge" and not item.bridge_subject
            for item in requirements
        ):
            raise ValueError(
                "graph bridge requirements require a canonical bridge_subject"
            )
        if any(
            item.role == "bridge" and item.bridge_kind is None
            for item in requirements
        ):
            raise ValueError(
                "graph bridge requirements require a validated bridge_kind"
            )
        requirement_by_id = {item.id: item for item in requirements}
        tasks = tuple(self.tasks)
        if len(tasks) > _MAX_TASKS:
            raise ValueError("retrieval task graph has too many tasks")
        if any(not isinstance(item, RetrievalTask) for item in tasks):
            raise ValueError("tasks must contain RetrievalTask values")
        task_by_id: dict[str, RetrievalTask] = {}
        for task in tasks:
            if task.task_id in task_by_id:
                raise ValueError("graph contains duplicate task ids")
            task_by_id[task.task_id] = task

            for requirement_id in task.target_requirement_ids:
                requirement = requirement_by_id.get(requirement_id)
                if requirement is None:
                    raise ValueError(
                        "task targets a requirement that does not exist"
                    )
                if task.role != "anchor" and task.role != requirement.role:
                    raise ValueError(
                        "task role does not match targeted requirement role"
                    )
            for dependency_id in task.dependency_task_ids:
                if dependency_id == task.task_id:
                    raise ValueError("task cannot depend on itself")

        for task in tasks:
            for dependency_id in task.dependency_task_ids:
                if dependency_id not in task_by_id:
                    raise ValueError("task contains a dangling dependency id")

        self._validate_acyclic(task_by_id)
        for task in tasks:
            for dependency_id in task.dependency_task_ids:
                dependency_role = task_by_id[dependency_id].role
                if task.role == "anchor":
                    raise ValueError("anchor tasks cannot have dependencies")
                if task.role == "answer" and dependency_role == "answer":
                    raise ValueError(
                        "answer tasks may depend only on bridge or anchor tasks"
                    )
                if task.role == "bridge" and dependency_role not in {"anchor"}:
                    raise ValueError(
                        "bridge tasks may depend only on anchor tasks"
                    )
            if task.role == "answer":
                bridge_parent_ids = {
                    dependency_id
                    for dependency_id in task.dependency_task_ids
                    if task_by_id[dependency_id].role == "bridge"
                }
                if set(task.bridge_edge_modes) != bridge_parent_ids:
                    raise ValueError(
                        "answer task bridge edge modes must match its bridge dependencies"
                    )

        self._validate_requirement_bindings(
            requirements=requirements,
            task_by_id=task_by_id,
        )
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "tasks", tasks)

    @staticmethod
    def _validate_acyclic(task_by_id: dict[str, RetrievalTask]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("retrieval task graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency_id in task_by_id[task_id].dependency_task_ids:
                visit(dependency_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_by_id:
            visit(task_id)

    @staticmethod
    def _validate_requirement_bindings(
        *,
        requirements: tuple[AnswerRequirementV2, ...],
        task_by_id: dict[str, RetrievalTask],
    ) -> None:
        requirement_by_id = {item.id: item for item in requirements}
        answer_tasks_by_requirement: dict[str, list[RetrievalTask]] = {}
        bridge_tasks_by_requirement: dict[str, list[RetrievalTask]] = {}
        for task in task_by_id.values():
            if task.role == "answer":
                answer_tasks_by_requirement.setdefault(
                    task.target_requirement_ids[0], []
                ).append(task)
            elif task.role == "bridge":
                bridge_tasks_by_requirement.setdefault(
                    task.target_requirement_ids[0], []
                ).append(task)

        # Every required answer has one and only one owner.  Optional answer
        # requirements may be omitted, but if present they still have one
        # owner, avoiding ambiguous evidence attribution.
        for requirement in requirements:
            if requirement.role != "answer":
                continue
            owners = answer_tasks_by_requirement.get(requirement.id, [])
            if len(owners) > 1:
                raise ValueError("answer requirement has multiple answer tasks")
            if requirement.is_required_answer and len(owners) != 1:
                raise ValueError(
                    "every required answer requirement must have exactly one answer task"
                )
            if owners and owners[0].required != requirement.is_required_answer:
                raise ValueError("answer task required flag is inconsistent")

        # A bridge can be represented by at most one logical task.  Its
        # requiredness is derived only from hard proof consumers (or the
        # bridge requirement itself), never merely from an optional
        # augmentation consumer.
        for requirement in requirements:
            if requirement.role != "bridge":
                continue
            owners = bridge_tasks_by_requirement.get(requirement.id, [])
            if len(owners) > 1:
                raise ValueError("bridge requirement has multiple bridge tasks")
            proof_consumers: list[RetrievalTask] = []
            augmentation_consumers: list[RetrievalTask] = []
            for task in task_by_id.values():
                if task.role != "answer":
                    continue
                answer_requirement = requirement_by_id[
                    task.target_requirement_ids[0]
                ]
                if requirement.id in answer_requirement.proof_bridge_requirement_ids:
                    proof_consumers.append(task)
                if requirement.id in answer_requirement.augmentation_bridge_requirement_ids:
                    augmentation_consumers.append(task)
            if not proof_consumers and not augmentation_consumers:
                if owners:
                    raise ValueError("graph contains an unreferenced bridge task")
                continue
            if not owners:
                raise ValueError("answer bridge edge has no bridge task")
            expected_required = bool(
                requirement.importance == "required"
                or any(task.required for task in proof_consumers)
            )
            if owners[0].required != expected_required:
                raise ValueError("bridge task required flag is inconsistent")

        # Each answer task must depend on exactly the bridge tasks declared by
        # its requirement, with the same proof/augmentation mode.  Anchor
        # dependencies are permitted for recall but deliberately excluded from
        # this semantic comparison.
        for task in task_by_id.values():
            if task.role != "answer":
                continue
            requirement = requirement_by_id[task.target_requirement_ids[0]]
            declared_bridge_modes = {
                **{
                    requirement_id: "proof"
                    for requirement_id in requirement.proof_bridge_requirement_ids
                },
                **{
                    requirement_id: "augmentation"
                    for requirement_id in requirement.augmentation_bridge_requirement_ids
                },
            }
            actual_bridge_modes = {
                task_by_id[dependency_id].target_requirement_ids[0]: (
                    task.bridge_edge_modes[dependency_id]
                )
                for dependency_id in task.dependency_task_ids
                if task_by_id[dependency_id].role == "bridge"
            }
            if actual_bridge_modes != declared_bridge_modes:
                raise ValueError(
                    "answer task bridge edge modes do not match its requirement"
                )

    @property
    def task_by_id(self) -> dict[str, RetrievalTask]:
        """Return a defensive lookup copy; callers cannot mutate the graph."""

        return {task.task_id: task for task in self.tasks}

    def answer_bridge_parent_task_ids(
        self,
        task_id: object,
        *,
        mode: BridgeEdgeMode | None = None,
    ) -> tuple[str, ...]:
        """Return typed bridge parents for one answer task.

        Execution code must use this rather than looking only at task role;
        that is what prevents optional augmentation edges from becoming proof
        blockers during a future scheduler or ledger change.
        """

        normalized_task_id = _text(task_id, field="task_id", max_chars=80)
        task = self.task_by_id.get(normalized_task_id)
        if task is None:
            raise ValueError("unknown task id in bridge parent lookup")
        if task.role != "answer":
            raise ValueError("bridge parent lookup requires an answer task")
        return task.bridge_parent_task_ids(mode=mode)

    def requirement_ids_reachable_from(
        self,
        task_ids: tuple[str, ...] | list[str] | set[str],
    ) -> frozenset[str]:
        """Return requirement owners in the task's downstream lineage.

        Retrieval provenance identifies the operation that returned a chunk;
        it is not itself evidence.  A root/anchor hit can therefore be tested
        against the answer tasks that descend from that root, while the
        lexical/evidence predicates still decide whether it actually supports
        those requirements.  This makes the anchor boundary explicit instead
        of treating an anchor as an answer owner.
        """

        task_by_id = self.task_by_id
        normalized_ids = tuple(dict.fromkeys(str(value) for value in task_ids))
        if any(value not in task_by_id for value in normalized_ids):
            raise ValueError("unknown task id in lineage lookup")
        reverse: dict[str, set[str]] = {task_id: set() for task_id in task_by_id}
        for task in self.tasks:
            for dependency_id in task.dependency_task_ids:
                reverse[dependency_id].add(task.task_id)
        pending = list(normalized_ids)
        visited: set[str] = set()
        result: set[str] = set()
        while pending:
            task_id = pending.pop()
            if task_id in visited:
                continue
            visited.add(task_id)
            task = task_by_id[task_id]
            result.update(task.target_requirement_ids)
            pending.extend(reverse[task_id] - visited)
        return frozenset(result)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph snapshot without full requirement text."""

        return {
            "schema_version": self.schema_version,
            "requirements": list(_requirement_snapshot(self.requirements)),
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def answer_bridge_paths(
        self,
        *,
        mode: BridgeEdgeMode,
    ) -> tuple[AnswerBridgePath, ...]:
        """Return explicitly declared bridge paths in stable task order.

        A caller must choose ``proof`` or ``augmentation``.  Leaving the
        mode implicit is precisely how the previous executor accidentally
        treated all bridge parents as one dependency family.
        """

        normalized_mode = str(mode or "").strip().casefold()
        if normalized_mode not in _BRIDGE_EDGE_MODES:
            raise ValueError("bridge path mode must be proof or augmentation")
        task_by_id = self.task_by_id
        paths: list[AnswerBridgePath] = []
        for task in self.tasks:
            if task.role != "answer":
                continue
            selected_bridge_task_ids = task.bridge_parent_task_ids(
                mode=normalized_mode,  # type: ignore[arg-type]
            )
            if not selected_bridge_task_ids:
                continue
            # An augmentation route still has to carry every proof parent of
            # the same answer.  Otherwise it could make an optional
            # classification fact look like permission to bypass a hard
            # condition/mapping prerequisite.
            bridge_task_ids = (
                task.bridge_parent_task_ids(mode="proof")
                if normalized_mode == "augmentation"
                else ()
            ) + selected_bridge_task_ids
            paths.append(AnswerBridgePath(
                answer_task_id=task.task_id,
                answer_requirement_id=task.target_requirement_ids[0],
                bridge_task_ids=bridge_task_ids,
                bridge_requirement_ids=tuple(
                    task_by_id[bridge_task_id].target_requirement_ids[0]
                    for bridge_task_id in bridge_task_ids
                ),
                edge_mode=normalized_mode,  # type: ignore[arg-type]
                bridge_edge_modes={
                    bridge_task_id: task.bridge_edge_modes[bridge_task_id]
                    for bridge_task_id in bridge_task_ids
                },
            ))
        return tuple(paths)

    def safe_summary(self) -> dict[str, Any]:
        """Return content-light diagnostics suitable for production traces."""

        counts = {role: 0 for role in ("answer", "bridge", "anchor")}
        bridge_edge_counts = {mode: 0 for mode in ("proof", "augmentation")}
        required_count = 0
        edge_count = 0
        for task in self.tasks:
            counts[task.role] += 1
            required_count += int(task.required)
            edge_count += len(task.dependency_task_ids)
            for mode in task.bridge_edge_modes.values():
                bridge_edge_counts[mode] += 1
        return {
            "schema_version": self.schema_version,
            "task_count": len(self.tasks),
            "task_counts": counts,
            "required_task_count": required_count,
            "edge_count": edge_count,
            "bridge_edge_counts": bridge_edge_counts,
            "requirement_count": len(self.requirements),
        }


@dataclass(frozen=True)
class RagExecutionBundle:
    """One immutable plan-to-DAG handoff for a single RAG execution.

    A query plan and a task graph are meaningful only as an exact pair.  This
    contract prevents the API from replacing a plan after graph compilation or
    from passing a graph generated for a different requirement set.  A plan
    that cannot produce a typed graph is explicitly ``not_ready``; it must be
    clarified or recompiled before it can touch retrieval.
    """

    plan: QueryPlanV2
    mode: ExecutionMode
    reason: str
    task_graph: RetrievalTaskGraph | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, QueryPlanV2):
            raise ValueError("execution bundle requires a QueryPlanV2")
        if self.mode not in _EXECUTION_MODES:
            raise ValueError("execution bundle mode is not supported")
        reason = _text(self.reason, field="execution bundle reason", max_chars=200)
        graph = self.task_graph
        if self.mode == "ledgered":
            if not isinstance(graph, RetrievalTaskGraph):
                raise ValueError("ledgered execution bundle requires a task graph")
            if tuple(graph.requirements) != tuple(self.plan.requirements):
                raise ValueError("task graph requirements must match execution plan")
            if self.plan.needs_clarification or self.plan.answer_shape == "unknown":
                raise ValueError("ledgered execution bundle requires a ready plan")
            if any(
                item.role == "bridge" and item.bridge_kind is None
                for item in self.plan.requirements
            ):
                raise ValueError("ledgered execution bundle requires typed bridge semantics")
        else:
            if graph is not None:
                raise ValueError("non-ledgered execution bundles cannot carry a task graph")
            # ``not_ready`` covers both an ordinary unresolved user request
            # and a structurally unsafe plan (for example, an untyped bridge
            # relation).  Neither may be coerced into a flat retrieval path.
            if not (
                self.plan.needs_clarification
                or self.plan.answer_shape == "unknown"
                or any(
                    item.role == "bridge" and item.bridge_kind is None
                    for item in self.plan.requirements
                )
            ):
                raise ValueError("not_ready execution bundle requires an unresolved plan")
        object.__setattr__(self, "reason", reason)

    @property
    def uses_task_ledger(self) -> bool:
        return self.mode == "ledgered"

    def safe_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "plan_schema_version": self.plan.schema_version,
            "answer_shape": self.plan.answer_shape,
            "requirement_count": len(self.plan.requirements),
            "task_graph": (
                self.task_graph.safe_summary()
                if self.task_graph is not None
                else None
            ),
        }


def _bridge_task_query(requirement: AnswerRequirementV2) -> str:
    subject = str(requirement.bridge_subject or "").strip()
    if not subject:
        raise ValueError("bridge requirement must define bridge_subject")
    bridge_kind = str(requirement.bridge_kind or "").strip().casefold()
    suffix = _BRIDGE_QUERY_SUFFIX_BY_KIND.get(bridge_kind)
    if suffix is None:
        raise ValueError("bridge requirement must define a validated bridge_kind")
    # Scope values are copied from the trusted requirement fields.  They are
    # constraints, not generated business vocabulary; no string is assembled
    # from an answer target or an inferred factual value.
    scope_terms: list[str] = []
    scope = requirement.applicability_scope
    for value in (
        scope.product if scope is not None else None,
        scope.version if scope is not None else None,
        (
            scope.project
            if scope is not None and scope.has_project_constraint
            else None
        ),
    ):
        normalized = str(value or "").strip()
        if normalized and normalized.casefold() not in {
            item.casefold() for item in scope_terms
        }:
            scope_terms.append(normalized)
    return " ".join((*scope_terms, subject, suffix))


def compile_retrieval_task_graph(plan: QueryPlanV2) -> RetrievalTaskGraph:
    """Compile a stable retrieval DAG from typed requirements only.

    The flat ``plan.retrieval_queries`` field is intentionally ignored.  This
    function therefore remains correct if query ordering changes or two tasks
    happen to carry equal query strings.
    """

    if not isinstance(plan, QueryPlanV2):
        raise ValueError("plan must be a QueryPlanV2")
    if plan.needs_clarification:
        raise ValueError("cannot compile a graph for a clarification plan")
    if plan.answer_shape == "unknown":
        raise ValueError("cannot compile a graph for an unknown plan")

    requirements = tuple(plan.requirements)
    bridge_requirements = tuple(
        item for item in requirements if item.role == "bridge"
    )
    bridge_task_id_by_requirement = {
        requirement.id: f"bridge_{requirement.id}"
        for requirement in bridge_requirements
    }

    anchor = RetrievalTask(
        task_id="anchor_root",
        role="anchor",
        query=plan.original_query,
    )
    bridge_tasks = tuple(
        RetrievalTask(
            task_id=bridge_task_id_by_requirement[requirement.id],
            role="bridge",
            query=_bridge_task_query(requirement),
            target_requirement_ids=(requirement.id,),
            dependency_task_ids=("anchor_root",),
            applicability_scope=requirement.applicability_scope,
            required=(
                requirement.importance == "required"
                or any(
                    answer.is_required_answer
                    and requirement.id in answer.proof_bridge_requirement_ids
                    for answer in requirements
                    if answer.role == "answer"
                )
            ),
        )
        for requirement in bridge_requirements
    )
    answer_tasks = tuple(
        RetrievalTask(
            task_id=f"answer_{requirement.id}",
            role="answer",
            query=requirement.description,
            target_requirement_ids=(requirement.id,),
            dependency_task_ids=tuple(
                (
                    "anchor_root",
                    *(
                        bridge_task_id_by_requirement[dependency_id]
                        for dependency_id in requirement.proof_bridge_requirement_ids
                    ),
                    *(
                        bridge_task_id_by_requirement[dependency_id]
                        for dependency_id
                        in requirement.augmentation_bridge_requirement_ids
                    ),
                )
            ),
            bridge_edge_modes={
                **{
                    bridge_task_id_by_requirement[dependency_id]: "proof"
                    for dependency_id in requirement.proof_bridge_requirement_ids
                },
                **{
                    bridge_task_id_by_requirement[dependency_id]: "augmentation"
                    for dependency_id
                    in requirement.augmentation_bridge_requirement_ids
                },
            },
            applicability_scope=requirement.applicability_scope,
            required=requirement.is_required_answer,
        )
        for requirement in requirements
        if requirement.role == "answer"
    )
    # Topological presentation order is stable: recall anchor, bridge roots,
    # then answer tasks.  No query-string deduplication occurs here.
    return RetrievalTaskGraph(
        requirements=requirements,
        tasks=(anchor, *bridge_tasks, *answer_tasks),
    )


def compile_rag_execution_bundle(plan: QueryPlanV2) -> RagExecutionBundle:
    """Compile the only plan/graph handoff accepted by the production runner."""

    if not isinstance(plan, QueryPlanV2):
        raise ValueError("plan must be a QueryPlanV2")
    if plan.needs_clarification:
        return RagExecutionBundle(
            plan=plan,
            mode="not_ready",
            reason="plan_needs_clarification",
        )
    if plan.answer_shape == "unknown":
        return RagExecutionBundle(
            plan=plan,
            mode="not_ready",
            reason="plan_answer_shape_unknown",
        )
    if any(
        item.role == "bridge" and item.bridge_kind is None
        for item in plan.requirements
    ):
        return RagExecutionBundle(
            plan=plan,
            mode="not_ready",
            reason="untyped_bridge_semantics",
        )
    return RagExecutionBundle(
        plan=plan,
        mode="ledgered",
        reason="all_bridge_semantics_validated",
        task_graph=compile_retrieval_task_graph(plan),
    )


__all__ = [
    "RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION",
    "AnswerBridgePath",
    "BridgeEdgeMode",
    "RagExecutionBundle",
    "RetrievalTask",
    "RetrievalTaskGraph",
    "compile_rag_execution_bundle",
    "compile_retrieval_task_graph",
]
