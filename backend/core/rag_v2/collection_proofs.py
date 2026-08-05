"""Pure, source-structural proofs for exhaustive collection answers.

Both evidence assembly and the final coverage graph need to decide whether a
source has actually closed a collection.  Keeping that decision in one pure
module prevents a renderer hint, a score, or a caller-supplied metadata flag
from becoming a second and weaker definition of "complete".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from core.rag_v2.bridge_resolution import (
    bridge_dependency_ids_for_answer,
    bridge_subject_for_requirement,
    content_matches_complete_answer_target,
)
from core.rag_v2.config_claims import matching_config_assignments
from core.rag_v2.contracts import AnswerRequirementV2, EvidenceItem


_COLLECTION_QUERY_PREFIX_RE = re.compile(
    r"^(?:(?:请问|请|麻烦|帮我(?:查|看|列|说明)?(?:一下)?)\s*|"
    r"(?:列出|列举|罗列|说明|概述|介绍)\s*|"
    r"(?:如何|怎么)(?:完成|执行|办理|操作|实现)?\s*|"
    r"(?:what\s+(?:is|are)\s+(?:the\s+)?|list\s+(?:all\s+)?(?:the\s+)?))",
    re.IGNORECASE,
)
_COLLECTION_QUERY_SUFFIX_RE = re.compile(
    r"(?:是什么|是哪些|(?:都)?有哪些(?:内容|项目|元素|成员|种类|方式|步骤|"
    r"要求|标准|规则|措施|选项)?|有(?:哪些|什么)(?:内容|项目|元素|成员|"
    r"种类|方式|步骤|要求|标准|规则|措施|选项)?|"
    r"(?:包括|包含)(?:哪些|什么)(?:内容|项目|元素|成员|种类|方式|步骤|"
    r"要求|标准|规则|措施|选项)?|分别是什么|(?:完整)?(?:清单|列表))$",
    re.IGNORECASE,
)
_COLLECTION_STRONG_CLOSURE_RE = re.compile(
    r"(?:仅(?:包括|包含|有|限于|支持|允许)|仅有|只有|"
    r"(?:共|一共|总计|合计)\s*(?:\d+|[一二三四五六七八九十百两]+)\s*"
    r"(?:项|条|个|种|类|步|阶段|部分|方面)|"
    r"全部(?:包括|包含|为|是|如下)|"
    r"所有(?:内容|项目|条目|成员|元素|项|条)?(?:如下|为|是)|"
    r"(?:由|是由)(?=.{1,200}(?:构成|组成))|"
    r"分为(?=.{1,200}(?:[、,，;；。]|$))|"
    r"(?:consists?\s+of|comprises?|only\s+(?:includes?|contains?)))",
    re.IGNORECASE,
)
_COLLECTION_INTRO_CLOSURE_RE = re.compile(
    r"(?:如下(?:所示)?|以下(?:为|是)|依次为|分别为|"
    r"(?:the\s+)?following)",
    re.IGNORECASE,
)
_COLLECTION_NON_EXHAUSTIVE_RE = re.compile(
    r"(?:^|[^仅])(?:包括|包含)|例如|诸如|之一|部分|主要|至少|等等|\betc\.?\b|"
    r"(?:\.\.\.|……)",
    re.IGNORECASE,
)
_COLLECTION_NUMBERED_ITEM_RE = re.compile(
    r"(?:^\s*|(?<=[：:；;。]))(?:步骤\s*)?"
    r"(?:\d+|[一二三四五六七八九十百]+)\s*[、.．):：]\s*\S+",
    re.MULTILINE,
)
_COLLECTION_REDIRECT_RE = re.compile(
    r"(?:另见|详见|参见|请见|请查看|"
    r"(?:查看|查阅)\s*(?:附件|附录|文档|详情|具体(?:内容|条款|规则)?|下文|表)|"
    r"(?:具体|详情|完整(?:内容|条款|规则)?)\s*(?:请)?(?:见|查看|参见)|"
    r"见(?:附件|附录|下文|制度(?:正文)?|文档|表)|"
    r"(?:附件|附录|下文|制度(?:正文)?|文档|表)\s*(?:中|内|详见))",
    re.IGNORECASE,
)
_NUMBERED_LINE_RE = re.compile(
    r"^\s*(?:步骤\s*)?(?:\d+|[一二三四五六七八九十百]+)\s*[、.．):：]\s*\S+"
)
_BULLET_LINE_RE = re.compile(r"^\s*[-*+]\s+\S+")
_EXCLUSIVE_CLOSURE_RE = re.compile(
    r"(?:仅(?:包括|包含|有|限于|支持|允许)|仅有|只有|only\s+(?:includes?|contains?))",
    re.IGNORECASE,
)
_OPEN_ENDED_MEMBER_RE = re.compile(
    r"(?:等等|等(?=[。；;，,、\s]|$)|"
    r"(?:及|以及)?其他(?:方式|项目|项|内容|规则|条款|情况)?|"
    r"例如|诸如|主要|至少|\.\.\.|……)",
    re.IGNORECASE,
)
_PROCEDURE_ARROW_SPLIT_RE = re.compile(r"\s*(?:→|->|=>|⟶|➔)\s*")
_PROCEDURE_INLINE_SPLIT_RE = re.compile(r"(?:、|[,，;；])")
_PROCEDURE_SOURCE_MARKER_RE = re.compile(
    r"(?:流程|步骤|顺序|依次|操作(?:步骤|流程)?|procedure|process)",
    re.IGNORECASE,
)
_PROCEDURE_SEQUENCE_CONNECTOR_RE = re.compile(
    r"(?:先|然后|再|随后|最后|依次|按(?:以下)?顺序)",
    re.IGNORECASE,
)
_PROCEDURE_DEFERRED_CONTINUATION_RE = re.compile(
    r"(?:"
    r"(?:后续|其余|剩余|未尽|后文|下一阶段)"
    r"[^。；;！？!?]{0,40}?"
    r"(?:另行|待|补充|详见|参见|见(?:附件|附录|下文|文档|表))"
    r"|(?:另行|待)\s*(?:处理|补充|确认|审批|安排|说明)"
    r")",
    re.IGNORECASE,
)


def collection_target_description(description: str) -> str:
    """Remove collection-question grammar while retaining its business target."""

    original = re.sub(r"\s+", " ", str(description or "")).strip()
    value = original.strip(" \t，,。；;：:！!？?")
    for _ in range(3):
        updated = _COLLECTION_QUERY_PREFIX_RE.sub("", value).strip()
        updated = _COLLECTION_QUERY_SUFFIX_RE.sub("", updated).strip()
        updated = updated.strip(" \t，,。；;：:！!？?")
        if updated == value:
            break
        value = updated
    return value if len(re.sub(r"\s+", "", value)) >= 2 else original


def _collection_bridge_subjects(
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> tuple[str, ...]:
    bridge_by_id = {
        value.id: value for value in requirements if value.role == "bridge"
    }
    return tuple(dict.fromkeys(
        subject
        for dependency_id in bridge_dependency_ids_for_answer(
            requirement,
            requirements,
        )
        for bridge in (bridge_by_id.get(dependency_id),)
        if bridge is not None
        if (subject := bridge_subject_for_requirement(bridge))
    ))


def collection_target_matches(
    requirement: AnswerRequirementV2,
    content: str,
    *,
    requirements: Sequence[AnswerRequirementV2],
) -> bool:
    """Check a source-local target anchor; scores and metadata never count."""

    normalized_content = re.sub(r"\s+", " ", str(content or "")).strip()
    if not normalized_content:
        return False
    return content_matches_complete_answer_target(
        collection_target_description(requirement.description),
        normalized_content,
        bridge_subjects=_collection_bridge_subjects(requirement, requirements),
    )


def table_local_scope_text(items: Sequence[EvidenceItem]) -> str:
    """Return parser-local table labels, deliberately excluding document title."""

    values: list[str] = []
    for item in items:
        raw_path = item.metadata.get("section_path")
        if isinstance(raw_path, (list, tuple)):
            values.extend(
                str(value).strip()
                for value in raw_path[1:]
                if str(value or "").strip()
            )
        for line in item.content.splitlines():
            normalized = line.strip()
            if (
                normalized.count("|") >= 2
                and not re.fullmatch(r"\|?[\s:|.-]+\|?", normalized)
            ):
                values.append(normalized.strip(" |"))
                break
    return "\n".join(dict.fromkeys(values))


def table_matches_collection_target(
    items: Sequence[EvidenceItem],
    *,
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> bool:
    return collection_target_matches(
        requirement,
        table_local_scope_text(items),
        requirements=requirements,
    )


ClosureKind = Literal["collection", "procedure"]


@dataclass(frozen=True)
class SourceSpan:
    """One exact source interval used by a declaration proof.

    Collection closure is intentionally a local proposition.  Keeping offsets
    (rather than only normalised strings) makes the boundary inspectable and
    prevents a later, unrelated sentence in the same chunk from being treated
    as a hidden qualifier of an earlier declaration.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise ValueError("source span offsets must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ValueError("source span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must be non-empty and ordered")

    def text(self, content: str) -> str:
        return str(content or "")[self.start:self.end]


@dataclass(frozen=True)
class SourceCollectionClosureProof:
    """A target-bound source declaration and its immediately attached members.

    ``attached_qualifier_spans`` is deliberately local: it contains only a
    redirect, open-ended enumeration, or deferred continuation parsed within
    the anchor declaration or its adjacent member block.  It never scans the
    entire chunk.  Consumers accept a proof only when :attr:`is_closed` is
    true, but retaining the intervals makes a rejected candidate explainable
    in diagnostics without treating arbitrary later text as part of it.
    """

    closure_kind: ClosureKind
    anchor_span: SourceSpan
    member_block_span: SourceSpan
    attached_qualifier_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.closure_kind not in {"collection", "procedure"}:
            raise ValueError("unsupported source closure kind")
        if self.member_block_span.start < self.anchor_span.start:
            raise ValueError("member block cannot precede its declaration")
        qualifiers = tuple(self.attached_qualifier_spans)
        if any(not isinstance(span, SourceSpan) for span in qualifiers):
            raise ValueError("attached qualifiers must be source spans")
        object.__setattr__(self, "attached_qualifier_spans", qualifiers)

    @property
    def is_closed(self) -> bool:
        return not self.attached_qualifier_spans


@dataclass(frozen=True)
class _SourceDeclaration:
    """A sentence-or-line declaration, preserving its original offsets."""

    span: SourceSpan
    text: str


_DECLARATION_TERMINATOR_RE = re.compile(r"[。！？!?]+")
_DECLARATION_EDGE_CHARS = " \t\r#|【】"
_MEMBER_EDGE_CHARS = " \t\r，,。；;：:！!？?"


def _closure_subject(unit: str, marker_start: int) -> str:
    prefix = unit[:marker_start].strip(" \t，,。；;：:")
    for separator in ("：", ":"):
        if separator not in prefix:
            continue
        parent, local = prefix.rsplit(separator, 1)
        # A local facet after a colon must not inherit an unrelated broad title.
        prefix = local.strip() or parent.strip()
        break
    return prefix


def _inline_member_count(tail: str) -> int:
    """Count authored members that occur after the closure marker itself.

    This primitive intentionally has no whole-chunk semantics.  Its caller
    supplies a declaration-local or member-block-local span and separately
    decides whether a parsed qualifier invalidates that exact declaration.
    """

    normalized = tail.strip(_MEMBER_EDGE_CHARS)
    if not normalized:
        return 0
    numbered_count = len(_COLLECTION_NUMBERED_ITEM_RE.findall(normalized))
    if numbered_count >= 2:
        return numbered_count
    members = [
        value.strip(" \t，,。；;：:")
        for value in re.split(r"(?:、|[,，;；]|(?:以及|和|及|与))", normalized)
        if value.strip(" \t，,。；;：:")
    ]
    return len(members)


def _trim_source_span(
    content: str,
    start: int,
    end: int,
    *,
    edge_chars: str,
) -> SourceSpan | None:
    """Trim a local span without moving it across a declaration boundary."""

    source = str(content or "")
    start = max(0, int(start))
    end = min(len(source), int(end))
    while start < end and source[start] in edge_chars:
        start += 1
    while end > start and source[end - 1] in edge_chars:
        end -= 1
    return SourceSpan(start, end) if start < end else None


def _iter_physical_line_spans(content: str, *, start: int = 0) -> tuple[SourceSpan, ...]:
    """Return exact physical lines from ``start`` onward, excluding newlines."""

    source = str(content or "")
    cursor = max(0, int(start))
    spans: list[SourceSpan] = []
    while cursor < len(source):
        newline = source.find("\n", cursor)
        end = len(source) if newline < 0 else newline
        if end > cursor and source[end - 1:end] == "\r":
            end -= 1
        if end > cursor:
            spans.append(SourceSpan(cursor, end))
        if newline < 0:
            break
        cursor = newline + 1
    return tuple(spans)


def _iter_declarations(content: str) -> tuple[_SourceDeclaration, ...]:
    """Split only sentence/line boundaries, never commas or semicolons.

    A comma or semicolon can introduce a qualifier of the same declaration;
    splitting on it would let ``仅包括 A、B，具体详见附件`` incorrectly close.
    Conversely, a later sentence must remain independently scoped.
    """

    source = str(content or "")
    declarations: list[_SourceDeclaration] = []
    for line_span in _iter_physical_line_spans(source):
        line = line_span.text(source)
        cursor = line_span.start
        for match in _DECLARATION_TERMINATOR_RE.finditer(line):
            raw_end = line_span.start + match.end()
            span = _trim_source_span(
                source,
                cursor,
                raw_end,
                edge_chars=_DECLARATION_EDGE_CHARS,
            )
            if span is not None:
                declarations.append(_SourceDeclaration(span, span.text(source)))
            cursor = raw_end
        span = _trim_source_span(
            source,
            cursor,
            line_span.end,
            edge_chars=_DECLARATION_EDGE_CHARS,
        )
        if span is not None:
            declarations.append(_SourceDeclaration(span, span.text(source)))
    return tuple(declarations)


def _span_matches(span: SourceSpan, pattern: re.Pattern[str], content: str) -> tuple[SourceSpan, ...]:
    """Project matches in a local source span back to absolute offsets."""

    result: list[SourceSpan] = []
    for match in pattern.finditer(span.text(content)):
        if match.end() <= match.start():
            continue
        result.append(SourceSpan(span.start + match.start(), span.start + match.end()))
    return tuple(result)


def _attached_qualifier_spans(
    content: str,
    spans: Sequence[SourceSpan],
    *,
    procedure: bool,
) -> tuple[SourceSpan, ...]:
    """Find only qualifiers parsed inside a declaration/member block.

    This function is the central scope boundary for closure safety.  A later
    chunk sentence is deliberately invisible unless the parser included it in
    one of ``spans``.  Overlapping regex matches are coalesced so diagnostics
    remain stable and do not turn repeated wording into multiple conditions.
    """

    patterns: tuple[re.Pattern[str], ...] = (
        _COLLECTION_REDIRECT_RE,
        _COLLECTION_NON_EXHAUSTIVE_RE,
        _OPEN_ENDED_MEMBER_RE,
        *((_PROCEDURE_DEFERRED_CONTINUATION_RE,) if procedure else ()),
    )
    matches: list[SourceSpan] = []
    for span in spans:
        for pattern in patterns:
            matches.extend(_span_matches(span, pattern, content))
    matches.sort(key=lambda value: (value.start, value.end))
    result: list[SourceSpan] = []
    for span in matches:
        if result and span.start <= result[-1].end:
            if span.end > result[-1].end:
                result[-1] = SourceSpan(result[-1].start, span.end)
            continue
        result.append(span)
    return tuple(result)


def _declaration_prefix_qualifier_spans(
    content: str,
    anchor_span: SourceSpan,
    *,
    before: int,
    procedure: bool,
) -> tuple[SourceSpan, ...]:
    """Inspect only the target-side prefix before its closure marker.

    Applying the open-ended matcher to the whole declaration would make the
    strong marker ``全部包括`` self-invalidating because it contains ``包括``.
    The marker/result is instead inspected by the member-block parser.  This
    prefix check still catches an explicitly attached qualifier such as
    ``采购流程（详见附件）如下`` without widening the proof to later prose.
    """

    prefix = _trim_source_span(
        content,
        anchor_span.start,
        before,
        edge_chars=_DECLARATION_EDGE_CHARS,
    )
    if prefix is None:
        return ()
    return _attached_qualifier_spans(
        content,
        (prefix,),
        procedure=procedure,
    )


def _member_prefix_before_qualifier(
    content: str,
    member_span: SourceSpan,
    qualifiers: Sequence[SourceSpan],
) -> SourceSpan | None:
    """Keep a diagnostic member span before its first attached qualifier."""

    end = min(
        (span.start for span in qualifiers if span.start >= member_span.start),
        default=member_span.end,
    )
    return _trim_source_span(
        content,
        member_span.start,
        end,
        edge_chars=_MEMBER_EDGE_CHARS,
    )


def _next_line_start(content: str, span: SourceSpan) -> int:
    newline = str(content or "").find("\n", span.end)
    return len(str(content or "")) if newline < 0 else newline + 1


def _following_local_member_block(
    content: str,
    anchor_span: SourceSpan,
    *,
    min_members: int,
    procedure: bool,
) -> tuple[SourceSpan, tuple[SourceSpan, ...]] | None:
    """Return only the immediately adjacent authored list/table block.

    A list elsewhere in the chunk is never allowed to close a declaration.
    Qualifiers occurring inside that contiguous block are explicitly attached;
    later prose after the block is intentionally outside the proof.
    """

    source = str(content or "")
    # An adjacent list belongs to a declaration only when nothing else was
    # authored after that declaration on the same physical line.  Otherwise a
    # later sentence can accidentally donate an unrelated attachment list.
    line_end = source.find("\n", anchor_span.end)
    if line_end >= 0 and source[anchor_span.end:line_end].strip(" \t\r"):
        return None
    lines = _iter_physical_line_spans(
        content,
        start=_next_line_start(content, anchor_span),
    )
    index = 0
    while index < len(lines) and not lines[index].text(content).strip():
        index += 1
    if index >= len(lines):
        return None

    first = lines[index]
    first_text = first.text(content)
    member_spans: list[SourceSpan] = []
    if _NUMBERED_LINE_RE.match(first_text) or _BULLET_LINE_RE.match(first_text):
        while index < len(lines):
            line = lines[index]
            line_text = line.text(content)
            if not (_NUMBERED_LINE_RE.match(line_text) or _BULLET_LINE_RE.match(line_text)):
                break
            member_spans.append(line)
            index += 1
    elif first_text.count("|") >= 2 and index + 1 < len(lines):
        separator = lines[index + 1].text(content).strip()
        if not re.fullmatch(r"\|?[\s:|.-]+\|?", separator):
            return None
        # Include header/separator in the local block so a redirect in a table
        # heading cannot be silently ignored.  Only non-separator rows count.
        member_spans.extend((first, lines[index + 1]))
        index += 2
        data_rows = 0
        while index < len(lines) and lines[index].text(content).count("|") >= 2:
            line = lines[index]
            member_spans.append(line)
            if not re.fullmatch(r"\|?[\s:|.-]+\|?", line.text(content).strip()):
                data_rows += 1
            index += 1
        if data_rows < min_members:
            return None
    else:
        return None

    if len(member_spans) < min_members:
        return None
    block = SourceSpan(member_spans[0].start, member_spans[-1].end)
    qualifiers = _attached_qualifier_spans(
        content,
        member_spans,
        procedure=procedure,
    )
    return block, qualifiers


def _inline_member_proof(
    *,
    content: str,
    anchor_span: SourceSpan,
    member_span: SourceSpan,
    min_members: int,
    closure_kind: ClosureKind,
) -> SourceCollectionClosureProof | None:
    qualifiers = _attached_qualifier_spans(
        content,
        (member_span,),
        procedure=closure_kind == "procedure",
    )
    effective_member_span = _member_prefix_before_qualifier(
        content,
        member_span,
        qualifiers,
    )
    if effective_member_span is None:
        return None
    if _inline_member_count(effective_member_span.text(content)) < min_members:
        return None
    return SourceCollectionClosureProof(
        closure_kind=closure_kind,
        anchor_span=anchor_span,
        member_block_span=effective_member_span,
        attached_qualifier_spans=qualifiers,
    )


def _following_member_proof(
    *,
    content: str,
    anchor_span: SourceSpan,
    anchor_qualifiers: Sequence[SourceSpan],
    min_members: int,
    closure_kind: ClosureKind,
) -> SourceCollectionClosureProof | None:
    located = _following_local_member_block(
        content,
        anchor_span,
        min_members=min_members,
        procedure=closure_kind == "procedure",
    )
    if located is None:
        return None
    member_span, member_qualifiers = located
    return SourceCollectionClosureProof(
        closure_kind=closure_kind,
        anchor_span=anchor_span,
        member_block_span=member_span,
        attached_qualifier_spans=tuple(
            dict.fromkeys((*anchor_qualifiers, *member_qualifiers))
        ),
    )


def _closure_kind_for_requirement(
    requirement: AnswerRequirementV2,
) -> ClosureKind | None:
    """Return the compiler-declared closure semantics for one requirement.

    This module proves source structure; it must not reinterpret the user
    wording to decide whether a request is a finite member list or an ordered
    procedure.  In particular, ``如何配置`` and ``配置项有哪些`` can share
    vocabulary while requiring different proof rules.  The semantic compiler
    records that decision as ``coverage_contract`` and this function only
    executes it.
    """

    contract = requirement.effective_coverage_contract
    if contract == "structured_collection":
        return "collection"
    if contract == "ordered_steps":
        return "procedure"
    return None


def _has_explicit_inline_procedure_sequence(
    result: str,
    *,
    subject: str,
) -> bool:
    """Validate a complete, source-authored inline sequence of steps."""

    normalized = result.strip(" \t，,。；;：:")
    if not normalized:
        return False

    arrow_steps = [
        value.strip(" \t，,。；;：:")
        for value in _PROCEDURE_ARROW_SPLIT_RE.split(normalized)
        if value.strip(" \t，,。；;：:")
    ]
    if len(arrow_steps) >= 2:
        return True

    numbered_steps = _COLLECTION_NUMBERED_ITEM_RE.findall(normalized)
    if len(numbered_steps) >= 2:
        return True

    # A bare ``A、B、C`` is an unordered member list, not proof that the
    # author intended A before B before C.  It becomes an ordered sequence
    # only when the source declaration itself carries procedure/step grammar
    # or the result includes an explicit sequencing connector.
    if not (
        _PROCEDURE_SOURCE_MARKER_RE.search(subject)
        or _PROCEDURE_SEQUENCE_CONNECTOR_RE.search(normalized)
    ):
        return False
    inline_steps = [
        value.strip(" \t，,。；;：:")
        for value in _PROCEDURE_INLINE_SPLIT_RE.split(normalized)
        if value.strip(" \t，,。；;：:")
    ]
    return len(inline_steps) >= 2


def _procedure_closure_candidates(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> tuple[SourceCollectionClosureProof, ...]:
    """Derive local procedure candidates without scanning later prose.

    A complete procedure needs a target-bound source label and either a local
    list/table block, an explicit arrow chain, or an inline multi-step
    sequence.  Deferred continuations invalidate only the declaration or
    adjacent list in which they occur; a later, unrelated sentence must not
    erase a closed procedure declaration.
    """

    if _closure_kind_for_requirement(requirement) != "procedure":
        return ()
    content = str(item.content or "")
    candidates: list[SourceCollectionClosureProof] = []

    def append_candidate(
        declaration: _SourceDeclaration,
        *,
        marker_start: int,
        marker_end: int,
        subject: str,
    ) -> None:
        """Append one procedure proof from a target-bound source marker."""

        if not collection_target_matches(
            requirement,
            subject,
            requirements=requirements,
        ):
            return
        result_span = _trim_source_span(
            content,
            declaration.span.start + marker_end,
            declaration.span.end,
            edge_chars=_MEMBER_EDGE_CHARS,
        )
        anchor_qualifiers = _declaration_prefix_qualifier_spans(
            content,
            declaration.span,
            before=declaration.span.start + marker_start,
            procedure=True,
        )
        if result_span is not None:
            proof = _inline_member_proof(
                content=content,
                anchor_span=declaration.span,
                member_span=result_span,
                min_members=2,
                closure_kind="procedure",
            )
            if proof is None or not _has_explicit_inline_procedure_sequence(
                proof.member_block_span.text(content),
                subject=subject,
            ):
                return
            candidates.append(SourceCollectionClosureProof(
                closure_kind=proof.closure_kind,
                anchor_span=proof.anchor_span,
                member_block_span=proof.member_block_span,
                attached_qualifier_spans=tuple(dict.fromkeys(
                    (*anchor_qualifiers, *proof.attached_qualifier_spans)
                )),
            ))
            return
        proof = _following_member_proof(
            content=content,
            anchor_span=declaration.span,
            anchor_qualifiers=anchor_qualifiers,
            min_members=2,
            closure_kind="procedure",
        )
        if proof is not None:
            candidates.append(proof)

    for declaration in _iter_declarations(content):
        separator_match = re.search(r"[：:]", declaration.text)
        if separator_match is not None:
            append_candidate(
                declaration,
                marker_start=separator_match.start(),
                marker_end=separator_match.end(),
                subject=declaration.text[:separator_match.start()].strip(),
            )
        # A procedure can also be authoritatively declared with an exhaustive
        # marker (``流程仅包括 A、B、C``) or an introduced adjacent step list
        # (``流程如下：\n1...``).  These are still procedure proofs because
        # the member grammar is checked above, not generic list certificates.
        for marker in _COLLECTION_STRONG_CLOSURE_RE.finditer(declaration.text):
            append_candidate(
                declaration,
                marker_start=marker.start(),
                marker_end=marker.end(),
                subject=_closure_subject(declaration.text, marker.start()),
            )
        for marker in _COLLECTION_INTRO_CLOSURE_RE.finditer(declaration.text):
            append_candidate(
                declaration,
                marker_start=marker.start(),
                marker_end=marker.end(),
                subject=_closure_subject(declaration.text, marker.start()),
            )
    return tuple(dict.fromkeys(candidates))


def has_explicit_procedure_closure(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> bool:
    """Return whether one target-bound procedure candidate is structurally closed."""

    return any(
        proof.is_closed
        for proof in _procedure_closure_candidates(
            item,
            requirement=requirement,
            requirements=requirements,
        )
    )


def derive_source_collection_closure_proofs(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> tuple[SourceCollectionClosureProof, ...]:
    """Derive candidate proofs from declaration-local source structure.

    The function intentionally returns rejected candidates too (with
    ``attached_qualifier_spans``) so callers can inspect exactly what made a
    declaration non-exhaustive.  It does not accept metadata hints or search
    scores.  The boolean compatibility helpers below accept only candidates
    whose local proof has no attached qualifier.
    """

    content = str(item.content or "")
    candidates: list[SourceCollectionClosureProof] = []
    closure_kind = _closure_kind_for_requirement(requirement)
    if closure_kind == "collection":
        # Configuration answers are often authored as a YAML/properties block
        # without numbered steps or an exhaustive prose marker.  The typed
        # assignment extractor is the source declaration in that format: it
        # binds the user's configuration target to concrete path/value spans
        # and therefore provides the same closure guarantee as a source-local
        # ``以下为`` list.  This is intentionally generic; it does not know
        # product names, parameter names or secret values.
        assignments = matching_config_assignments(
            requirement.description,
            content,
        )
        if assignments:
            first = min(assignments, key=lambda value: value.source_start)
            last = max(assignments, key=lambda value: value.source_end)
            candidates.append(SourceCollectionClosureProof(
                closure_kind="collection",
                anchor_span=SourceSpan(
                    start=first.source_start,
                    end=first.source_end,
                ),
                member_block_span=SourceSpan(
                    start=first.source_start,
                    end=last.source_end,
                ),
            ))
    # An ordered procedure is not merely an unordered collection.  Its own
    # closure path below has stricter sequence grammar, so do not emit a
    # second, weaker ``collection`` certificate for the same evidence.
    if closure_kind == "collection":
        for declaration in _iter_declarations(content):
            for match in _COLLECTION_STRONG_CLOSURE_RE.finditer(declaration.text):
                subject = _closure_subject(declaration.text, match.start())
                if not collection_target_matches(
                    requirement,
                    subject,
                    requirements=requirements,
                ):
                    continue
                member_span = _trim_source_span(
                    content,
                    declaration.span.start + match.end(),
                    declaration.span.end,
                    edge_chars=_MEMBER_EDGE_CHARS,
                )
                anchor_qualifiers = _declaration_prefix_qualifier_spans(
                    content,
                    declaration.span,
                    before=declaration.span.start + match.start(),
                    procedure=False,
                )
                min_members = 1 if _EXCLUSIVE_CLOSURE_RE.search(match.group(0)) else 2
                if member_span is not None:
                    proof = _inline_member_proof(
                        content=content,
                        anchor_span=declaration.span,
                        member_span=member_span,
                        min_members=min_members,
                        closure_kind="collection",
                    )
                    if proof is not None:
                        candidates.append(SourceCollectionClosureProof(
                            closure_kind=proof.closure_kind,
                            anchor_span=proof.anchor_span,
                            member_block_span=proof.member_block_span,
                            attached_qualifier_spans=tuple(dict.fromkeys(
                                (*anchor_qualifiers, *proof.attached_qualifier_spans)
                            )),
                        ))
                    continue
                proof = _following_member_proof(
                    content=content,
                    anchor_span=declaration.span,
                    anchor_qualifiers=anchor_qualifiers,
                    min_members=min_members,
                    closure_kind="collection",
                )
                if proof is not None:
                    candidates.append(proof)

            for match in _COLLECTION_INTRO_CLOSURE_RE.finditer(declaration.text):
                subject = _closure_subject(declaration.text, match.start())
                if not collection_target_matches(
                    requirement,
                    subject,
                    requirements=requirements,
                ):
                    continue
                member_span = _trim_source_span(
                    content,
                    declaration.span.start + match.end(),
                    declaration.span.end,
                    edge_chars=_MEMBER_EDGE_CHARS,
                )
                anchor_qualifiers = _declaration_prefix_qualifier_spans(
                    content,
                    declaration.span,
                    before=declaration.span.start + match.start(),
                    procedure=False,
                )
                if member_span is not None:
                    proof = _inline_member_proof(
                        content=content,
                        anchor_span=declaration.span,
                        member_span=member_span,
                        min_members=2,
                        closure_kind="collection",
                    )
                    if proof is not None:
                        candidates.append(SourceCollectionClosureProof(
                            closure_kind=proof.closure_kind,
                            anchor_span=proof.anchor_span,
                            member_block_span=proof.member_block_span,
                            attached_qualifier_spans=tuple(dict.fromkeys(
                                (*anchor_qualifiers, *proof.attached_qualifier_spans)
                            )),
                        ))
                    continue
                proof = _following_member_proof(
                    content=content,
                    anchor_span=declaration.span,
                    anchor_qualifiers=anchor_qualifiers,
                    min_members=2,
                    closure_kind="collection",
                )
                if proof is not None:
                    candidates.append(proof)

    if closure_kind == "procedure":
        candidates.extend(_procedure_closure_candidates(
            item,
            requirement=requirement,
            requirements=requirements,
        ))
    return tuple(dict.fromkeys(candidates))


def has_explicit_collection_closure(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> bool:
    """Prove one collection only from its source text and requirement target.

    A title, a retrieval label, a previously inferred assertion, or an
    arbitrary ``structural_collection_closed`` metadata flag cannot close the
    collection.  In particular, a local ``交通如下`` clause never inherits a
    broader ``公司出差标准`` title just because it occurs in the same chunk.
    """

    return any(
        proof.is_closed
        for proof in derive_source_collection_closure_proofs(
            item,
            requirement=requirement,
            requirements=requirements,
        )
    )


__all__ = [
    "collection_target_description",
    "collection_target_matches",
    "derive_source_collection_closure_proofs",
    "has_explicit_collection_closure",
    "has_explicit_procedure_closure",
    "SourceCollectionClosureProof",
    "SourceSpan",
    "table_local_scope_text",
    "table_matches_collection_target",
]
