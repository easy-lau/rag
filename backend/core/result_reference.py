"""Shared surface contract for ordinal result references.

The parser recognizes only language structure such as ``the fourth item`` or
``the last document``.  It never resolves an ordinal to a document identity;
that authority remains with a request-local result catalog or an already
re-authorized active task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ResultReferenceKind = Literal["ordinal", "last", "prefix"]

_ORDINAL_RE = re.compile(
    r"第(?P<ordinal>[0-9一二三四五六七八九十两]+)"
    r"(?:个|篇|份|条|章|节|款)?"
    r"(?:文章|文档|资料|文件)?",
    re.IGNORECASE,
)
_LAST_RE = re.compile(
    r"(?:最后|末尾)(?:一个|一篇|一份|一条)?(?:文章|文档|资料|文件)?",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(
    r"前(?P<count>[0-9一二三四五六七八九十两]+)(?:个|篇|份|条)?",
    re.IGNORECASE,
)

_RESULT_READ_VERB_RE = re.compile(
    r"(?:看|查看|打开|阅读|读|正文|内容|详情|显示|给我|我要|我想|返回)",
    re.IGNORECASE,
)
_CORRECTION_HINT_RE = re.compile(
    r"(?:不是|错了|错了|搞错|弄错|应该|才是|才对|纠正|更正|修正)",
    re.IGNORECASE,
)


def small_ordinal(value: object) -> int | None:
    """Parse a bounded Chinese or Arabic ordinal without business vocabulary."""

    text = str(value or "").strip()
    if text.isdigit():
        number = int(text)
        return number if 1 <= number <= 20 else None
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        units = digits.get(right, 0) if right else 0
        number = tens * 10 + units
        return number if 1 <= number <= 20 else None
    return digits.get(text)


@dataclass(frozen=True)
class ResultReferenceSurface:
    kind: ResultReferenceKind
    value: int | None
    span: str


def parse_result_reference_surface(value: object) -> ResultReferenceSurface | None:
    """Return one unambiguous ordinal selection expressed in source text."""

    text = str(value or "")
    matches: list[tuple[int, int, ResultReferenceKind, int | None, str]] = []
    for match in _ORDINAL_RE.finditer(text):
        ordinal = small_ordinal(match.group("ordinal"))
        if ordinal is not None:
            matches.append(
                (match.start(), match.end(), "ordinal", ordinal, match.group(0))
            )
    for match in _LAST_RE.finditer(text):
        matches.append((match.start(), match.end(), "last", None, match.group(0)))
    for match in _PREFIX_RE.finditer(text):
        count = small_ordinal(match.group("count"))
        if count is not None:
            matches.append(
                (match.start(), match.end(), "prefix", count, match.group(0))
            )
    matches.sort(key=lambda item: (item[0], item[1]))
    if len(matches) != 1:
        return None
    _, _, kind, parsed_value, span = matches[0]
    return ResultReferenceSurface(kind=kind, value=parsed_value, span=span)


def is_result_list_reference(question: object) -> bool:
    """Whether the question selects an item from a displayed result list.

    ``第N个/第N篇/最后一份/前N个`` are result-list selections.  ``第N条/第N章/
    第N节/第N款`` usually point at a regulation clause inside a document rather
    than at a numbered list item; they are treated as result references only
    when the question also reads like a correction or an explicit read request.
    """

    surface = parse_result_reference_surface(question)
    if surface is None:
        return False
    if surface.kind in {"last", "prefix"}:
        return True
    if surface.kind != "ordinal":
        return False
    if surface.span.endswith(("条", "章", "节", "款")):
        text = str(question or "")
        return bool(
            _RESULT_READ_VERB_RE.search(text)
            or _CORRECTION_HINT_RE.search(text)
        )
    return True


def same_result_reference_selection(left: object, right: object) -> bool:
    """Whether two turns contain the same single ordinal result selection."""

    left_reference = parse_result_reference_surface(left)
    right_reference = parse_result_reference_surface(right)
    return bool(
        left_reference is not None
        and right_reference is not None
        and left_reference.kind == right_reference.kind
        and left_reference.value == right_reference.value
    )


__all__ = [
    "ResultReferenceSurface",
    "is_result_list_reference",
    "parse_result_reference_surface",
    "same_result_reference_selection",
    "small_ordinal",
]
