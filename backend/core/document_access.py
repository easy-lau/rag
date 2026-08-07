"""Document object-level authorization policy.

Knowledge-base scope is enforced by ``require_kb_access`` before a document is
loaded.  This module owns the remaining object decision so response hints and
mutation enforcement cannot drift into two different authorization rules.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from core.permissions import DOC_DELETE, DOC_READ, DOC_UPDATE
from models.db_models import Document, User


DocumentAction = Literal["read", "update", "delete"]


@dataclass(frozen=True)
class DocumentPermissions:
    read: bool
    update: bool
    delete: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


class DocumentAccessDenied(PermissionError):
    """A domain-level denial that API adapters can audit and translate."""

    def __init__(self, action: DocumentAction, reason: str):
        super().__init__(reason)
        self.action = action
        self.reason = reason


def _has_permission(user: User, permission: str) -> bool:
    return bool(user.is_superadmin or permission in getattr(user, "permissions", ()))


def is_document_owner(user: User, document: Document) -> bool:
    """Missing owners are intentionally not claimable by ordinary users."""

    return bool(document.created_by is not None and document.created_by == user.id)


def evaluate_document_permissions(user: User, document: Document) -> DocumentPermissions:
    """Evaluate per-document actions after the containing KB scope is accepted."""

    if user.is_superadmin:
        return DocumentPermissions(read=True, update=True, delete=True)

    is_owner = is_document_owner(user, document)
    # 草稿只对创建者（及超管）可见：非本人既不能查看，也不能出现在列表/统计中；
    # 正式入库（status=ready）后才恢复为普通文档的可读规则。
    is_draft = str(getattr(document, "status", "") or "").strip().casefold() == "draft"
    return DocumentPermissions(
        read=_has_permission(user, DOC_READ) and (is_owner or not is_draft),
        update=is_owner and _has_permission(user, DOC_UPDATE),
        delete=is_owner and _has_permission(user, DOC_DELETE),
    )


def _denial_reason(user: User, document: Document, action: DocumentAction) -> str:
    required = {"read": DOC_READ, "update": DOC_UPDATE, "delete": DOC_DELETE}[action]
    if not _has_permission(user, required):
        return "missing_capability"
    if action != "read" and document.created_by is None:
        return "document_owner_missing"
    return "not_document_owner"


def require_document_action(
    user: User,
    document: Document,
    action: DocumentAction,
) -> None:
    """Require one action using exactly the same decision returned to clients."""

    permissions = evaluate_document_permissions(user, document)
    if getattr(permissions, action):
        return
    raise DocumentAccessDenied(action, _denial_reason(user, document, action))
