"""文档入库：解析 -> 切块 -> 向量化 -> 落库。"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from anyio import to_thread

from app.config import Settings
from app.core.chunking import build_chunks
from app.core.embedding import Embedder
from app.core.parsing import parse_bytes
from app.core.stores import Store
from app.core.types import Chunk, DocumentRecord


@dataclass(slots=True)
class IngestResult:
    document: DocumentRecord
    duplicate: bool
    chunk_count: int
    elapsed_ms: int


class IngestionService:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    async def ingest(self, filename: str, data: bytes, tenant: str = "default") -> IngestResult:
        started = time.perf_counter()

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise ValueError(f"文件超过 {self.settings.max_upload_mb}MB 限制")
        if not data:
            raise ValueError("文件为空")

        content_hash = hashlib.sha256(data).hexdigest()

        # 同一份文件重复上传直接返回，省钱也省时间
        existing = await to_thread.run_sync(self.store.find_by_hash, tenant, content_hash)
        if existing:
            return IngestResult(
                document=existing,
                duplicate=True,
                chunk_count=existing.chunk_count,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        text = parse_bytes(filename, data)
        drafts = build_chunks(
            text,
            strategy=self.settings.chunk_strategy,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            parent_size=self.settings.parent_chunk_size,
            child_size=self.settings.child_chunk_size,
        )
        if not drafts:
            raise ValueError("切块结果为空")

        document_id = uuid.uuid4().hex[:24]
        chunk_ids = {
            draft.index: hashlib.sha256(f"{document_id}:{draft.index}".encode()).hexdigest()[:24]
            for draft in drafts
        }

        chunks = [
            Chunk(
                id=chunk_ids[draft.index],
                document_id=document_id,
                tenant=tenant,
                text=draft.text,
                index=draft.index,
                parent_id=(
                    chunk_ids[draft.parent_local_id]
                    if draft.parent_local_id is not None
                    else None
                ),
                meta={
                    **draft.meta,
                    "filename": filename,
                    "document_id": document_id,
                },
            )
            for draft in drafts
        ]

        # 只给子块算向量：父块只用于拼上下文，不需要进入检索空间
        children = [chunk for chunk in chunks if chunk.role == "child"]
        vectors = await self.embedder.embed_documents([chunk.text for chunk in children])
        for chunk, vector in zip(children, vectors):
            chunk.embedding = vector

        document = DocumentRecord(
            id=document_id,
            tenant=tenant,
            filename=filename,
            content_hash=content_hash,
            size_bytes=len(data),
            chunk_count=len(children),
            created_at=datetime.now(timezone.utc),
        )
        await to_thread.run_sync(self.store.add_document, document, chunks)

        return IngestResult(
            document=document,
            duplicate=False,
            chunk_count=len(children),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
