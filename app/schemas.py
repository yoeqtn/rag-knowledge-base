"""对外 API 的请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool = True


class CitationOut(BaseModel):
    index: int
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    cache_hit: bool
    latency_ms: int
    usage: UsageOut
    cost_cny: float


class DocumentOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


class IngestResponse(BaseModel):
    document: DocumentOut
    duplicate: bool
    chunk_count: int
    elapsed_ms: int


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentOut]
