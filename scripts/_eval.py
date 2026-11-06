"""两个对比实验共用的检索评测逻辑。
抽出来单独放，是因为"怎么算指标"必须唯一。
如果切块实验和召回实验各写一份，两次跑的 MRR 口径可能悄悄不一致，
结论就没法互相印证了。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from anyio import to_thread

from app.core.parsing import SUPPORTED_EXTENSIONS
from app.core.retrieval import Retriever
from app.core.stores import Store
from app.services.ingest import IngestionService


async def reset_tenant(store: Store, tenant: str) -> None:
    """清空一个实验租户，避免重跑时累加旧数据。"""
    documents = await to_thread.run_sync(store.list_documents, tenant, 1000, 0)
    for document in documents:
        await to_thread.run_sync(store.delete_document, document.id, tenant)


async def ingest_dir(
    store: Store, embedder, settings, docs_dir: Path, tenant: str
) -> int:
    """把目录下的文档全部导入指定租户，返回新增的子块数。"""
    service = IngestionService(store, embedder, settings)
    files = [
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    chunks = 0
    for path in files:
        result = await service.ingest(path.name, path.read_bytes(), tenant)
        if not result.duplicate:
            chunks += result.chunk_count
    return chunks


@dataclass(slots=True)
class RetrievalMetrics:
    questions: int
    recall: float
    mrr: float
    context_hit: float
    keyword_coverage: float
    precision: float
    avg_context_chars: float
    avg_latency_ms: float
    details: list[dict] = field(default_factory=list)


async def evaluate_retrieval(
    retriever: Retriever,
    items: list[dict],
    tenant: str,
    top_k: int,
    query_vectors: Sequence[list[float]],
    warmup: bool = True,
) -> RetrievalMetrics:
    # 预热一次：BM25 倒排索引在第一次检索时才构建，
    # 不预热的话这部分一次性开销会混进第一个问题的延迟里
    if warmup and items:
        await retriever.retrieve(
            items[0]["question"],
            tenant,
            top_k=top_k,
            query_vector=query_vectors[0] if query_vectors else None,
        )

    hits = 0
    mrr = 0.0
    context_chars = 0
    latencies: list[float] = []
    context_hits = 0
    keyword_scores: list[float] = []
    precisions: list[float] = []
    details: list[dict] = []

    for item, query_vector in zip(items, query_vectors):
        expected = item.get("expected_source")
        started = time.perf_counter()
        sources = await retriever.retrieve(
            item["question"], tenant, top_k=top_k, query_vector=query_vector
        )
        latencies.append((time.perf_counter() - started) * 1000)

        filenames = [source.chunk.meta.get("filename") for source in sources]
        rank = filenames.index(expected) + 1 if expected in filenames else None
        if rank:
            hits += 1
            mrr += 1.0 / rank

        context_chars += sum(len(source.text_for_prompt) for source in sources)

        keywords = item.get("expected_keywords") or []
        context_hit: bool | None = None
        precision: float | None = None

        if keywords:
            context_text = "\n".join(source.text_for_prompt for source in sources)
            matched = sum(1 for keyword in keywords if keyword in context_text)
            keyword_scores.append(matched / len(keywords))
            context_hit = matched == len(keywords)
            if context_hit:
                context_hits += 1

            # 用子块原文算精度，不用回溯后的父块，否则对 parent_child 有偏
            if sources:
                useful = sum(
                    1
                    for source in sources
                    if any(keyword in source.chunk.text for keyword in keywords)
                )
                precision = useful / len(sources)
                precisions.append(precision)

        details.append(
            {
                "question": item["question"],
                "rank": rank,
                "context_hit": context_hit,
                "precision": precision,
            }
        )

    total = len(items)
    keyword_total = len(keyword_scores)
    return RetrievalMetrics(
        questions=total,
        recall=hits / total if total else 0.0,
        mrr=mrr / total if total else 0.0,
        context_hit=context_hits / keyword_total if keyword_total else 0.0,
        keyword_coverage=sum(keyword_scores) / keyword_total if keyword_total else 0.0,
        precision=sum(precisions) / len(precisions) if precisions else 0.0,
        avg_context_chars=round(context_chars / total, 1) if total else 0.0,
        avg_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        details=details,
    )
