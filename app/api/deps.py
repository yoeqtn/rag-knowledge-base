"""FastAPI 依赖：取容器、取租户、校验 API Key。"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from app.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_tenant(x_tenant_id: str = Header(default="default", alias="X-Tenant-Id")) -> str:
    return x_tenant_id.strip() or "default"


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = request.app.state.container.settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="无效的 API Key")
