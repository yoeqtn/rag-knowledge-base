"""存储层。

两个实现，同一套接口：
- MemoryStore：进程内，零依赖，用于单元测试和离线演示
- PgStore    ：PostgreSQL + pgvector，生产用

上层（检索、问答）只依赖 Store 协议，换存储不需要改业务代码。
这就是面试里"依赖倒置"最直接的例子。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from app.config import Settings
from app.core.types import Chunk, DocumentRecord, QueryLog


class Store(Protocol):
    def add_document(self, document: DocumentRecord, chunks: list[Chunk]) -> None: ...

    def get_document(self, document_id: str, tenant: str) -> DocumentRecord | None: ...

    def find_by_hash(self, tenant: str, content_hash: str) -> DocumentRecord | None: ...

    def list_documents(
        self, tenant: str, limit: int = 50, offset: int = 0
    ) -> list[DocumentRecord]: ...

    def delete_document(self, document_id: str, tenant: str) -> bool: ...

    def vector_search(
        self, vector: list[float], tenant: str, top_k: int
    ) -> list[tuple[Chunk, float]]: ...

    def all_chunks(self, tenant: str, role: str = "child") -> list[Chunk]: ...

    def get_chunks_by_ids(self, chunk_ids: list[str], tenant: str) -> list[Chunk]: ...

    def log_query(self, record: QueryLog) -> None: ...

    def stats(self, tenant: str) -> dict: ...

    def version(self) -> int: ...

    def health(self) -> dict[str, str]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# 内存实现
# --------------------------------------------------------------------------- #
class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, Chunk] = {}
        self._by_tenant: dict[str, list[str]] = {}
        self._logs: list[QueryLog] = []
        self._version = 0
        self._matrix_cache: dict[tuple[int, str], tuple[list[str], np.ndarray]] = {}

    # ---------- 文档 ----------
    def add_document(self, document: DocumentRecord, chunks: list[Chunk]) -> None:
        self._documents[document.id] = document
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._by_tenant.setdefault(document.tenant, []).extend(c.id for c in chunks)
        self._version += 1

    def get_document(self, document_id: str, tenant: str) -> DocumentRecord | None:
        document = self._documents.get(document_id)
        return document if document and document.tenant == tenant else None

    def find_by_hash(self, tenant: str, content_hash: str) -> DocumentRecord | None:
        for document in self._documents.values():
            if document.tenant == tenant and document.content_hash == content_hash:
                return document
        return None

    def list_documents(
        self, tenant: str, limit: int = 50, offset: int = 0
    ) -> list[DocumentRecord]:
        docs = [d for d in self._documents.values() if d.tenant == tenant]
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs[offset : offset + limit]

    def delete_document(self, document_id: str, tenant: str) -> bool:
        document = self.get_document(document_id, tenant)
        if not document:
            return False
        self._documents.pop(document_id, None)
        for chunk_id in list(self._chunks):
            if self._chunks[chunk_id].document_id == document_id:
                self._chunks.pop(chunk_id)
        self._by_tenant[tenant] = [
            cid for cid in self._by_tenant.get(tenant, []) if cid in self._chunks
        ]
        self._version += 1
        return True

    # ---------- 检索 ----------
    def _matrix(self, tenant: str) -> tuple[list[str], np.ndarray]:
        key = (self._version, tenant)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached

        ids: list[str] = []
        vectors: list[list[float]] = []
        for chunk_id in self._by_tenant.get(tenant, []):
            chunk = self._chunks.get(chunk_id)
            if chunk and chunk.role == "child" and chunk.embedding is not None:
                ids.append(chunk_id)
                vectors.append(chunk.embedding)

        matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._matrix_cache = {key: (ids, matrix)}
        return ids, matrix

    def vector_search(
        self, vector: list[float], tenant: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        ids, matrix = self._matrix(tenant)
        if not ids:
            return []
        query = np.asarray(vector, dtype=np.float32)
        scores = matrix @ query  # 已归一化，点积即余弦
        order = np.argsort(-scores)[:top_k]
        return [(self._chunks[ids[i]], float(scores[i])) for i in order]

    def all_chunks(self, tenant: str, role: str = "child") -> list[Chunk]:
        return [
            self._chunks[cid]
            for cid in self._by_tenant.get(tenant, [])
            if cid in self._chunks and self._chunks[cid].role == role
        ]

    def get_chunks_by_ids(self, chunk_ids: list[str], tenant: str) -> list[Chunk]:
        found = {
            cid: self._chunks[cid]
            for cid in chunk_ids
            if cid in self._chunks and self._chunks[cid].tenant == tenant
        }
        return [found[cid] for cid in chunk_ids if cid in found]

    # ---------- 日志与统计 ----------
    def log_query(self, record: QueryLog) -> None:
        record.created_at = record.created_at or datetime.now(timezone.utc)
        self._logs.append(record)

    def stats(self, tenant: str) -> dict:
        logs = [log for log in self._logs if log.tenant == tenant]
        return _aggregate_logs(logs, self._count_documents(tenant), len(self.all_chunks(tenant)))

    def _count_documents(self, tenant: str) -> int:
        return sum(1 for d in self._documents.values() if d.tenant == tenant)

    def version(self) -> int:
        return self._version

    def health(self) -> dict[str, str]:
        return {"backend": "memory", "status": "ok"}

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# PostgreSQL + pgvector 实现
# --------------------------------------------------------------------------- #
class PgStore:
    def __init__(self, database_url: str, embedding_dim: int) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.core import pg_models  # 延迟导入，只有用 postgres 时才需要

        self._models = pg_models
        self.embedding_dim = embedding_dim
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._version = 0
        self.init_schema()

    def init_schema(self) -> None:
        from sqlalchemy import text

        pg_models = self._models
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        pg_models.Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            # HNSW 适合在线查询；数据量小时建不建都行，但写进简历要有
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )

    # ---------- 文档 ----------
    def add_document(self, document: DocumentRecord, chunks: list[Chunk]) -> None:
        models = self._models
        with self._Session() as session, session.begin():
            session.add(
                models.DocumentRow(
                    id=document.id,
                    tenant=document.tenant,
                    filename=document.filename,
                    content_hash=document.content_hash,
                    size_bytes=document.size_bytes,
                    chunk_count=document.chunk_count,
                    created_at=document.created_at,
                )
            )
            # 没有定义 relationship，SQLAlchemy 不知道 chunks 依赖 documents，
            # 必须手动 flush 保证父行先落库，否则外键会报错
            session.flush()
            session.add_all(
                [
                    models.ChunkRow(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        tenant=chunk.tenant,
                        text=chunk.text,
                        chunk_index=chunk.index,
                        parent_id=chunk.parent_id,
                        role=chunk.role,
                        meta=chunk.meta,
                        embedding=chunk.embedding,
                    )
                    for chunk in chunks
                ]
            )
        self._version += 1

    def get_document(self, document_id: str, tenant: str) -> DocumentRecord | None:
        models = self._models
        with self._Session() as session:
            row = session.get(models.DocumentRow, document_id)
            if not row or row.tenant != tenant:
                return None
            return _to_document(row)

    def find_by_hash(self, tenant: str, content_hash: str) -> DocumentRecord | None:
        from sqlalchemy import select

        models = self._models
        with self._Session() as session:
            row = session.scalar(
                select(models.DocumentRow).where(
                    models.DocumentRow.tenant == tenant,
                    models.DocumentRow.content_hash == content_hash,
                )
            )
            return _to_document(row) if row else None

    def list_documents(
        self, tenant: str, limit: int = 50, offset: int = 0
    ) -> list[DocumentRecord]:
        from sqlalchemy import select

        models = self._models
        with self._Session() as session:
            rows = session.scalars(
                select(models.DocumentRow)
                .where(models.DocumentRow.tenant == tenant)
                .order_by(models.DocumentRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return [_to_document(row) for row in rows]

    def delete_document(self, document_id: str, tenant: str) -> bool:
        from sqlalchemy import delete

        models = self._models
        with self._Session() as session, session.begin():
            row = session.get(models.DocumentRow, document_id)
            if not row or row.tenant != tenant:
                return False
            session.execute(
                delete(models.ChunkRow).where(models.ChunkRow.document_id == document_id)
            )
            session.delete(row)
        self._version += 1
        return True

    # ---------- 检索 ----------
    def vector_search(
        self, vector: list[float], tenant: str, top_k: int
    ) -> list[tuple[Chunk, float]]:
        from sqlalchemy import select

        models = self._models
        distance = models.ChunkRow.embedding.cosine_distance(vector)
        stmt = (
            select(models.ChunkRow, distance.label("distance"))
            .where(
                models.ChunkRow.tenant == tenant,
                models.ChunkRow.role == "child",
                models.ChunkRow.embedding.is_not(None),
            )
            .order_by(distance)
            .limit(top_k)
        )
        with self._Session() as session:
            rows = session.execute(stmt).all()
            return [(_to_chunk(row), 1.0 - float(dist)) for row, dist in rows]

    def all_chunks(self, tenant: str, role: str = "child") -> list[Chunk]:
        from sqlalchemy import select

        models = self._models
        with self._Session() as session:
            rows = session.scalars(
                select(models.ChunkRow).where(
                    models.ChunkRow.tenant == tenant,
                    models.ChunkRow.role == role,
                )
            ).all()
            return [_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str], tenant: str) -> list[Chunk]:
        from sqlalchemy import select

        if not chunk_ids:
            return []
        models = self._models
        with self._Session() as session:
            rows = session.scalars(
                select(models.ChunkRow).where(
                    models.ChunkRow.id.in_(chunk_ids),
                    models.ChunkRow.tenant == tenant,
                )
            ).all()
            found = {row.id: _to_chunk(row) for row in rows}
        return [found[cid] for cid in chunk_ids if cid in found]

    # ---------- 日志与统计 ----------
    def log_query(self, record: QueryLog) -> None:
        models = self._models
        with self._Session() as session, session.begin():
            session.add(
                models.QueryLogRow(
                    tenant=record.tenant,
                    question=record.question,
                    answer=record.answer,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    cost_cny=record.cost_cny,
                    latency_ms=record.latency_ms,
                    cache_hit=record.cache_hit,
                )
            )

    def stats(self, tenant: str) -> dict:
        from sqlalchemy import func, select

        models = self._models
        with self._Session() as session:
            documents = session.scalar(
                select(func.count()).select_from(models.DocumentRow).where(
                    models.DocumentRow.tenant == tenant
                )
            )
            chunks = session.scalar(
                select(func.count()).select_from(models.ChunkRow).where(
                    models.ChunkRow.tenant == tenant,
                    models.ChunkRow.role == "child",
                )
            )
            row = session.execute(
                select(
                    func.count(models.QueryLogRow.id),
                    func.coalesce(func.sum(models.QueryLogRow.prompt_tokens), 0),
                    func.coalesce(func.sum(models.QueryLogRow.completion_tokens), 0),
                    func.coalesce(func.sum(models.QueryLogRow.cost_cny), 0.0),
                    func.coalesce(func.avg(models.QueryLogRow.latency_ms), 0.0),
                ).where(models.QueryLogRow.tenant == tenant)
            ).one()
            cache_hits = session.scalar(
                select(func.count())
                .select_from(models.QueryLogRow)
                .where(
                    models.QueryLogRow.tenant == tenant,
                    models.QueryLogRow.cache_hit.is_(True),
                )
            )

        total, prompt_tokens, completion_tokens, cost, avg_latency = row
        return {
            "documents": int(documents or 0),
            "chunks": int(chunks or 0),
            "queries": int(total or 0),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "cost_cny": round(float(cost or 0.0), 6),
            "avg_latency_ms": round(float(avg_latency or 0.0), 1),
            "cache_hit_rate": round(float(cache_hits or 0) / total, 4) if total else 0.0,
        }

    def version(self) -> int:
        return self._version

    def health(self) -> dict[str, str]:
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"backend": "postgres", "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            return {"backend": "postgres", "status": f"error: {exc}"}

    def close(self) -> None:
        self.engine.dispose()


# --------------------------------------------------------------------------- #
# 工厂与转换
# --------------------------------------------------------------------------- #
def build_store(settings: Settings) -> Store:
    backend = settings.storage_backend.lower()
    if backend == "memory":
        return MemoryStore()
    if backend == "postgres":
        return PgStore(settings.database_url, settings.embedding_dim)
    raise ValueError(f"未知 STORAGE_BACKEND: {settings.storage_backend}")


def _aggregate_logs(logs: list[QueryLog], documents: int, chunks: int) -> dict:
    total = len(logs)
    return {
        "documents": documents,
        "chunks": chunks,
        "queries": total,
        "prompt_tokens": sum(log.prompt_tokens for log in logs),
        "completion_tokens": sum(log.completion_tokens for log in logs),
        "cost_cny": round(sum(log.cost_cny for log in logs), 6),
        "avg_latency_ms": round(sum(log.latency_ms for log in logs) / total, 1) if total else 0.0,
        "cache_hit_rate": round(sum(1 for log in logs if log.cache_hit) / total, 4) if total else 0.0,
    }


def _to_document(row) -> DocumentRecord:
    return DocumentRecord(
        id=row.id,
        tenant=row.tenant,
        filename=row.filename,
        content_hash=row.content_hash,
        size_bytes=row.size_bytes,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )


def _to_chunk(row) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        tenant=row.tenant,
        text=row.text,
        index=row.chunk_index,
        parent_id=row.parent_id,
        meta=row.meta or {},
        embedding=list(row.embedding) if row.embedding is not None else None,
    )
