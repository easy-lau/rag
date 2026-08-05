"""Deterministic extraction of source-authored configuration assignments.

Configuration evidence is structurally different from prose facts: the
answer is often a path/value declaration whose human meaning appears in an
inline comment.  Treating it as generic categorical prose either drops the
claim or mistakes container labels such as ``documentation`` for the answer.
This module keeps the declaration typed and source-bound.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


_CONFIG_INTENT_RE = re.compile(
    r"(?:配置|设置|参数|开关|开启|关闭|启用|禁用|修改|调整|怎么办|"
    r"如何处理|怎么处理|configur|setting|parameter|enable|disable)",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{1,127}")
_YAML_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_.-]{0,127})\s*:\s*"
    r"(?P<value>[^#\r\n]*?)(?:\s+#\s*(?P<comment>[^\r\n]*))?$"
)
_PROPERTY_LINE_RE = re.compile(
    r"^(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_.-]{1,127})\s*=\s*"
    r"(?P<value>[^#\r\n]+?)(?:\s+#\s*(?P<comment>[^\r\n]*))?$",
    re.IGNORECASE,
)
_INLINE_ASSIGNMENT_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]{1,127})\s*[:=]\s*"
    r"(?P<value>true|false|null|[-+]?\d+(?:\.\d+)?|"
    r"\"[^\"\r\n]{0,200}\"|'[^'\r\n]{0,200}')"
    r"(?:\s*#\s*(?P<comment>.*?))?"
    r"(?=(?:\s+[A-Za-z_][A-Za-z0-9_.-]{1,127}\s*[:=])|```|$)",
    re.IGNORECASE,
)
# Markdown importers used by the knowledge-base pipeline may collapse a YAML
# code block into one physical line.  The ordinary line parser then loses the
# mapping path and the inline assignment parser only recognises booleans and
# numbers.  Recover the bounded ``key: value`` token stream structurally; the
# parser never invents values or consults product-specific vocabulary.
_FENCED_BLOCK_RE = re.compile(r"```(?P<body>.*?)```", re.DOTALL)
_INLINE_MAPPING_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_.&-])"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.&-]{0,127})"
    r"\s*:\s*"
)
_CODE_LANGUAGE_RE = re.compile(
    r"^\s*(?:yaml|yml|properties|property|ini|conf|toml|bash|shell)"
    r"(?:\s+|\r?\n)",
    re.IGNORECASE,
)
_CHINESE_RE = re.compile(r"[\u3400-\u9fff]{2,}")


@dataclass(frozen=True)
class ConfigAssignmentClaim:
    path: tuple[str, ...]
    value: str | bool | int | float | None
    normalized_value: str
    meaning: str | None
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        if not self.path or any(not item for item in self.path):
            raise ValueError("configuration path must not be empty")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("configuration source span is invalid")

    @property
    def normalized_path(self) -> str:
        return ".".join(item.casefold() for item in self.path)

    @property
    def normalized_assignment(self) -> str:
        return f"{self.normalized_path}={self.normalized_value}"


def _parse_value(raw: str) -> tuple[str | bool | int | float | None, str]:
    text = raw.strip().rstrip(",")
    lowered = text.casefold()
    if lowered == "true":
        return True, "true"
    if lowered == "false":
        return False, "false"
    if lowered in {"null", "none", "~"}:
        return None, "null"
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text), str(int(text))
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        value = float(text)
        return value, format(value, "g")
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {'"', "'"}
    ):
        text = text[1:-1]
    return text, json.dumps(text, ensure_ascii=False, separators=(",", ":"))


def _clean_comment(value: object) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" `#，,。；;")
    return text[:240] or None


def _line_assignments(text: str) -> Iterable[ConfigAssignmentClaim]:
    stack: list[tuple[int, str]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        match = _YAML_LINE_RE.match(line)
        if match is not None:
            indent_text = match.group("indent").replace("\t", "  ")
            indent = len(indent_text)
            key = match.group("key")
            raw_value = match.group("value").strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if not raw_value:
                stack.append((indent, key))
            else:
                value, normalized = _parse_value(raw_value)
                yield ConfigAssignmentClaim(
                    path=tuple(item for _, item in stack) + (key,),
                    value=value,
                    normalized_value=normalized,
                    meaning=_clean_comment(match.group("comment")),
                    source_start=cursor + match.start("key"),
                    source_end=cursor + match.end(),
                )
            cursor += len(raw_line)
            continue
        match = _PROPERTY_LINE_RE.match(line.strip())
        if match is not None:
            value, normalized = _parse_value(match.group("value"))
            key = match.group("key")
            key_offset = line.find(key)
            yield ConfigAssignmentClaim(
                path=tuple(part for part in key.split(".") if part),
                value=value,
                normalized_value=normalized,
                meaning=_clean_comment(match.group("comment")),
                source_start=cursor + max(0, key_offset),
                source_end=cursor + len(line),
            )
        cursor += len(raw_line)


def _valid_inline_mapping_matches(text: str) -> list[re.Match[str]]:
    """Return mapping keys, excluding URL-like ``http:`` value fragments."""

    matches: list[re.Match[str]] = []
    for match in _INLINE_MAPPING_KEY_RE.finditer(text):
        tail = text[match.end():].lstrip()
        if tail.startswith("//") or tail.startswith("\\"):
            continue
        # A new mapping key is separated from the previous scalar by a space,
        # punctuation or the beginning of the block.  This also keeps a
        # colon embedded in an unquoted URL inside the value segment.
        if match.start() > 0 and not (
            text[match.start() - 1].isspace()
            or text[match.start() - 1] in "[{(,;"
        ):
            continue
        matches.append(match)
    return matches


def _inline_mapping_assignments(
    text: str,
    *,
    offset: int = 0,
) -> Iterable[ConfigAssignmentClaim]:
    """Recover assignments from a flattened YAML/properties code block.

    Indentation is unavailable after flattening, so a container encountered
    after a scalar closes the previous container level.  This gives stable
    paths for the common ``root: child: key: value sibling: ...`` form while
    keeping the extracted leaf/value itself authoritative.  Ambiguous later
    nesting is never used to decide whether a claim exists.
    """

    matches = _valid_inline_mapping_matches(text)
    stack: list[str] = []
    previous_was_leaf = False
    for position, match in enumerate(matches):
        key = match.group("key")
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        raw_value = text[match.end():end].strip()
        # Markdown fences and inline comments are delimiters, not values.
        raw_value = raw_value.split("```", 1)[0].strip()
        comment_match = re.search(r"\s+#\s*(?P<comment>.*)$", raw_value)
        meaning = _clean_comment(
            comment_match.group("comment") if comment_match is not None else None
        )
        if comment_match is not None:
            raw_value = raw_value[:comment_match.start()].strip()
        if not raw_value:
            if previous_was_leaf and stack:
                stack.pop()
            stack.append(key)
            previous_was_leaf = False
            continue
        value, normalized = _parse_value(raw_value)
        yield ConfigAssignmentClaim(
            # A flattened block no longer carries indentation, so retaining a
            # synthetic parent path would be more dangerous than preserving
            # the source-authored leaf key.  Standard multiline YAML still
            # gets its full path from ``_line_assignments`` above.
            path=(key,),
            value=value,
            normalized_value=normalized,
            meaning=meaning,
            source_start=offset + match.start("key"),
            source_end=offset + end,
        )
        previous_was_leaf = True


def _fenced_inline_assignments(text: str) -> Iterable[ConfigAssignmentClaim]:
    for block in _FENCED_BLOCK_RE.finditer(text):
        body = block.group("body")
        language = _CODE_LANGUAGE_RE.match(body)
        body_offset = block.start("body")
        if language is not None:
            body_offset += language.end()
            body = body[language.end():]
        # Preserve the indentation-aware line parser for normal multiline
        # blocks; the recovery scanner is only for importer-flattened bodies.
        if "\n" in body or "\r" in body:
            continue
        yield from _inline_mapping_assignments(body, offset=body_offset)


def extract_config_assignments(content: Any) -> tuple[ConfigAssignmentClaim, ...]:
    text = str(content or "")
    if not text.strip():
        return ()
    claims: list[ConfigAssignmentClaim] = list(_line_assignments(text))
    claims.extend(_fenced_inline_assignments(text))
    occupied = [(item.source_start, item.source_end) for item in claims]
    for match in _INLINE_ASSIGNMENT_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        value, normalized = _parse_value(match.group("value"))
        key = match.group("key")
        claims.append(ConfigAssignmentClaim(
            path=tuple(part for part in key.split(".") if part),
            value=value,
            normalized_value=normalized,
            meaning=_clean_comment(match.group("comment")),
            source_start=match.start(),
            source_end=match.end(),
        ))
    unique: list[ConfigAssignmentClaim] = []
    seen: set[tuple[str, str]] = set()
    for claim in sorted(claims, key=lambda item: (item.source_start, item.source_end)):
        identity = (
            claim.normalized_path,
            claim.normalized_value,
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(claim)
    return tuple(unique[:100])


def _chinese_bigrams(value: str) -> set[str]:
    result: set[str] = set()
    for segment in _CHINESE_RE.findall(value):
        result.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return result


def _config_assignment_match_score(
    question: str,
    claim: ConfigAssignmentClaim,
) -> float:
    """Require both configuration intent and a source-visible target match."""

    query = re.sub(r"\s+", "", str(question or "")).casefold()
    if not query or not _CONFIG_INTENT_RE.search(query):
        return 0.0
    identifiers = {
        item.casefold()
        for item in _IDENTIFIER_RE.findall(question)
    }
    path_tokens = {
        claim.normalized_path,
        claim.path[-1].casefold(),
    }
    if identifiers & path_tokens or any(token in query for token in path_tokens):
        return 100.0
    meaning = re.sub(r"\s+", "", claim.meaning or "").casefold()
    if not meaning:
        return 0.0
    if meaning in query or query in meaning:
        return 50.0 + len(_chinese_bigrams(meaning))
    meaning_bigrams = _chinese_bigrams(meaning)
    query_bigrams = _chinese_bigrams(query)
    overlap = meaning_bigrams & query_bigrams
    if (
        len(overlap) < 2
        or len(overlap) / max(len(meaning_bigrams), 1) < 0.4
    ):
        return 0.0
    return float(len(overlap)) + (
        len(overlap) / max(len(meaning_bigrams), 1)
    )


def config_assignment_matches_query(
    question: str,
    claim: ConfigAssignmentClaim,
) -> bool:
    return _config_assignment_match_score(question, claim) > 0


def matching_config_assignments(
    question: str,
    content: Any,
) -> tuple[ConfigAssignmentClaim, ...]:
    scored = [
        (claim, _config_assignment_match_score(question, claim))
        for claim in extract_config_assignments(content)
    ]
    maximum = max((score for _, score in scored), default=0.0)
    if maximum <= 0:
        return ()
    # In a dense configuration block several nearby comments can share generic
    # words such as "默认密码".  Keep a close group for broad questions so a
    # setting and its related enforcement switch are not silently split apart;
    # an exact identifier/meaning match (score >= 50) still wins alone.  The
    # tolerance is a score-policy constant, not a product-specific exception.
    floor = maximum if maximum >= 50 else max(2.0, maximum - 1.0)
    return tuple(claim for claim, score in scored if score >= floor)


__all__ = [
    "ConfigAssignmentClaim",
    "config_assignment_matches_query",
    "extract_config_assignments",
    "matching_config_assignments",
]
