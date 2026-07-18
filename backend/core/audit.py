"""操作审计：在写接口里记录关键动作。

用法：接口签名加 `audit: AuditLogger = Depends(get_audit)`，在 commit 前后调用
`audit.log(db, "user.create", target_type="user", target_id=..., target_name=..., detail={...})`。
log() 只是把一条 OperationLog 加入会话，随业务 commit 一起落库（原子）。

红线：detail 绝不写入密码 / API Key / Token 等敏感值，密钥类只记"哪些键被改"。
"""

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_user
from models.db_models import OperationLog, User


def client_ip(request: Request) -> str | None:
    """解析客户端 IP：优先取 X-Forwarded-For 首段，否则回退 request.client.host。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


class AuditLogger:
    """每请求一个实例，封装操作人与来源信息。"""

    def __init__(self, request: Request, user: User):
        self.user = user
        self.ip = client_ip(request)
        self.user_agent = request.headers.get("user-agent")

    def log(
        self,
        db: AsyncSession,
        action: str,
        *,
        target_type: str | None = None,
        target_id=None,
        target_name: str | None = None,
        detail: dict | None = None,
    ) -> None:
        """把一条操作日志加入会话（不单独 commit，随业务一起提交）。"""
        db.add(OperationLog(
            user_id=self.user.id if self.user else None,
            username=self.user.username if self.user else "",
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
            target_name=target_name,
            detail=detail,
            ip=self.ip,
            user_agent=self.user_agent,
        ))


async def get_audit(request: Request, user: User = Depends(get_current_user)) -> AuditLogger:
    """审计依赖：复用已鉴权的 current_user（FastAPI 同请求内缓存，不会重复解析）。"""
    return AuditLogger(request, user)
