"""跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ChunkDraft:
    """切块结果，尚未写入存储。

    parent_local_id 只在 parent_child 策略下使用，指向同一批次内父块的 index，
    写入存储时会替换成真实的父块 id。
    """

    text: str
    index: int
    parent_local_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    tenant: str
    text: str
    index: int
    parent_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def role(self) -> str:
        return str(self.meta.get("role", "child"))


@dataclass(slots=True)
class DocumentRecord:
    id: str
    tenant: str
    filename: str
    content_hash: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    matched_by: tuple[str, ...] = ()
    # parent_child 策略下，命中子块后回溯到的父块全文；没有父块时为 None
    context: str | None = None

    @property
    def text_for_prompt(self) -> str:
        return self.context or self.chunk.text


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class QueryLog:
    tenant: str
    question: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    cost_cny: float
    latency_ms: int
    cache_hit: bool
    created_at: datetime | None = None
