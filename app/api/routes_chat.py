"""问答接口：普通 JSON 与 SSE 流式两种。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, get_tenant
from app.container import Container
from app.core.llm import DisabledChatModel
from app.schemas import ChatRequest, ChatResponse, CitationOut, UsageOut

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> ChatResponse:
    _ensure_llm_ready(container)
    result = await container.qa.answer(
        question=request.question,
        tenant=tenant,
        history=[m.model_dump() for m in request.history],
        top_k=request.top_k,
        use_cache=request.use_cache,
    )
    return ChatResponse(
        answer=result.answer,
        citations=[CitationOut(**c) for c in result.citations],
        cache_hit=result.cache_hit,
        latency_ms=result.latency_ms,
        usage=UsageOut(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_cny=result.cost_cny,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> StreamingResponse:
    _ensure_llm_ready(container)

    async def event_source():
        try:
            async for event in container.qa.stream_answer(
                question=request.question,
                tenant=tenant,
                history=[m.model_dump() for m in request.history],
                top_k=request.top_k,
                use_cache=request.use_cache,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 让 Nginx 别缓冲
        },
    )


def _ensure_llm_ready(container: Container) -> None:
    if isinstance(container.llm, DisabledChatModel):
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 LLM_API_KEY，无法生成回答。请在 .env 中填写后重启。",
        )
