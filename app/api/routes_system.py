"""健康检查与统计。"""

from __future__ import annotations

from anyio import to_thread
from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_tenant
from app.container import Container
from app.core.llm import DisabledChatModel

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(container: Container = Depends(get_container)) -> dict:
    return {
        "status": "ok",
        "storage": await to_thread.run_sync(container.store.health),
        "embedder": {
            "provider": container.settings.embedding_provider,
            "dim": container.embedder.dim,
        },
        "llm": {
            "model": container.settings.llm_model,
            "configured": not isinstance(container.llm, DisabledChatModel),
        },
        "cache": container.cache.stats(),
    }


@router.get("/stats")
async def stats(
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> dict:
    data = await to_thread.run_sync(container.store.stats, tenant)
    data["semantic_cache"] = container.cache.stats()
    return data
