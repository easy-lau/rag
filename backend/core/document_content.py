"""Canonical document-content formatting shared by ingestion and answer rendering.

Documents arrive from Markdown, DOCX and browser exports with slightly different
line/block conventions.  The RAG index must search the same readable text that a
user sees, so formatting cleanup belongs in one boundary instead of being
reimplemented by each answer path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


DOCUMENT_CONTENT_FORMAT_VERSION = "markdown-canonical.v1"

_HTML_CODE_RE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.IGNORECASE | re.DOTALL)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_FENCED_BLOCK_RE = re.compile(
    r"```\s*([A-Za-z0-9_+.-]*)\s*(.*?)```",
    re.DOTALL,
)
_SECTION_CONTEXT_RE = re.compile(r"^【(?P<context>[^】]+)】\s*(?:\n|$)")
_KNOWN_FENCE_LANGUAGES = frozenset({
    "bash", "shell", "sh", "zsh", "fish", "powershell", "ps1",
    "python", "py", "javascript", "js", "typescript", "ts", "java",
    "kotlin", "go", "rust", "sql", "yaml", "yml", "json", "xml",
    "html", "css", "dockerfile", "toml", "ini", "text", "txt",
})


def _normalise_code_span(match: re.Match[str]) -> str:
    value = re.sub(r"\s+", " ", match.group(1).strip())
    if not value:
        return "``"
    return f"`{value}`"


def _expand_inline_fenced_blocks(text: str) -> str:
    """Put inline triple-backtick blocks on their own lines.

    Browser/editor exports frequently collapse a complete Markdown fence into
    one paragraph.  This function only changes fence delimiters; code content is
    never interpreted or rewritten.
    """

    def replace(match: re.Match[str]) -> str:
        language = match.group(1).strip()
        body = match.group(2).strip(" \t\r\n")
        if language.casefold() not in _KNOWN_FENCE_LANGUAGES:
            # Some exports omit the separator between the language and the
            # first command (for example `````shellmv ...`````); retaining the
            # token as code is safer than inventing a language or dropping it.
            body = " ".join(part for part in (language, body) if part).strip()
            language = ""
        opener = f"```{language}" if language else "```"
        return f"\n{opener}\n{body}\n```\n"

    return _FENCED_BLOCK_RE.sub(replace, text)


def _split_embedded_list_markers(text: str) -> str:
    """Recover list lines from collapsed prose without touching code blocks."""

    lines: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue

        if re.match(r"^\s*\*\s+", line):
            line = re.sub(r"^\s*\*\s+", "- ", line, count=1)

        # Only split repeated markers so ordinary multiplication and prose
        # asterisks remain untouched.  This is intentionally content-agnostic:
        # no article title or business phrase is used as a formatting rule.
        if len(re.findall(r"\s+\*\s+", line)) >= 2:
            line = re.sub(r"\s+\*\s+", "\n- ", line)
        lines.extend(line.splitlines() or [""])
    return "\n".join(lines)


def normalize_document_markdown(content: str | None) -> str:
    """Return canonical, readable Markdown while preserving document facts.

    This is deliberately a formatting normalizer, not a semantic rewriter: it
    does not add, remove, reorder or summarize factual content.
    """

    text = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ""
    text = _HTML_BREAK_RE.sub("\n", text)
    text = _HTML_CODE_RE.sub(_normalise_code_span, text)
    text = _expand_inline_fenced_blocks(text)
    text = _split_embedded_list_markers(text)
    # Keep intentional indentation inside fences, but collapse accidental
    # trailing whitespace and excessive blank lines outside them.
    output: list[str] = []
    in_fence = False
    blank_count = 0
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            blank_count += 1
            if blank_count > 2:
                continue
        else:
            blank_count = 0
        output.append(line)
    # Remove only empty boundary lines.  Leading indentation is meaningful for
    # Markdown indented code blocks and must not be stripped.
    return "\n".join(output).strip("\n").rstrip()


def _section_heading(chunk: Any, document_name: str) -> tuple[str | None, str | None]:
    metadata = getattr(chunk, "metadata_", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    path = metadata.get("section_path")
    if isinstance(path, (list, tuple)):
        values = [str(item).strip() for item in path if str(item).strip()]
        if values:
            context = " › ".join(values)
            heading = values[-1]
            if heading == document_name and len(values) > 1:
                heading = values[-2]
            return context, heading
    return None, None


def render_document_chunks(
    document_name: str,
    chunks: Iterable[Any],
    *,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """Render ordered chunks as readable Markdown for result-reference answers."""

    parts: list[str] = [f"## 《{str(document_name).strip()}》".strip()]
    seen_headings: set[str] = set()
    truncated = False
    for chunk in chunks:
        body = normalize_document_markdown(getattr(chunk, "content", ""))
        if not body:
            continue
        context, heading = _section_heading(chunk, str(document_name).strip())
        if context:
            prefix = f"【{context}】"
            if body.startswith(prefix):
                body = body[len(prefix):].lstrip(" \n")
        if heading and heading != str(document_name).strip() and heading not in seen_headings:
            parts.append(f"### {heading}")
            seen_headings.add(heading)
        if body:
            parts.append(body)

    rendered = "\n\n".join(part for part in parts if part).strip()
    if max_chars is not None and len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip()
        truncated = True
    return rendered, truncated


def strip_section_context(content: str | None) -> str:
    """Remove the retrieval-only ``【document › section】`` prefix from a chunk."""

    text = normalize_document_markdown(content)
    match = _SECTION_CONTEXT_RE.match(text)
    return text[match.end():].lstrip() if match else text


__all__ = [
    "DOCUMENT_CONTENT_FORMAT_VERSION",
    "normalize_document_markdown",
    "render_document_chunks",
    "strip_section_context",
]
