"""问答服务：检索 -> 拼 Prompt -> 调模型 -> 解析引用 -> 计费/缓存/日志。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from anyio import to_thread

from app.config import Settings
from app.core.cache import SemanticCache
from app.core.citations import extract_citations
from app.core.embedding import Embedder
from app.core.llm import ChatModel
from app.core.prompt import (
    NO_CONTEXT_ANSWER,
    build_messages,
    build_rewrite_messages,
)
from app.core.retrieval import Retriever
from app.core.retrieval import RetrievedChunk
from app.core.stores import Store
from app.core.types import QueryLog, Usage


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    cache_hit: bool = False
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    cost_cny: float = 0.0


class QAService:
    def __init__(
        self,
        store: Store,
        retriever: Retriever,
        llm: ChatModel,
        embedder: Embedder,
        cache: SemanticCache,
        settings: Settings,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        self.cache = cache
        self.settings = settings

    # ------------------------------------------------------------------ #
    # 非流式
    # ------------------------------------------------------------------ #
    async def answer(
        self,
        question: str,
        tenant: str = "default",
        history: Sequence[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> AnswerResult:
        started = time.perf_counter()

        search_query, query_vector, sources, cached = await self._prepare(
            question, tenant, history, top_k, use_cache
        )

        if cached is not None:
            result = AnswerResult(
                answer=cached.answer,
                citations=cached.citations,
                cache_hit=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await self._log(tenant, question, result)
            return result

        if not sources:
            result = AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await self._log(tenant, question, result)
            return result

        messages = build_messages(question, sources, history)
        answer, usage = await self.llm.complete(messages)
        citations = [c.to_dict() for c in extract_citations(answer, sources)]

        result = AnswerResult(
            answer=answer,
            citations=citations,
            sources=sources,
            usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost_cny=self._cost(usage),
        )
        if use_cache and not history:
            self.cache.put(query_vector, question, answer, citations)

        await self._log(tenant, question, result)
        return result

    # ------------------------------------------------------------------ #
    # 流式
    # ------------------------------------------------------------------ #
    async def stream_answer(
        self,
        question: str,
        tenant: str = "default",
        history: Sequence[dict[str, str]] | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[dict]:
        started = time.perf_counter()
        search_query, query_vector, sources, cached = await self._prepare(
            question, tenant, history, top_k, use_cache
        )

        # 先告诉前端检索到了什么，用户等待首字节时就有东西可看
        yield {"type": "sources", "sources": [self._preview(s, i) for i, s in enumerate(sources, 1)]}

        if cached is not None:
            result = AnswerResult(
                answer=cached.answer,
                citations=cached.citations,
                cache_hit=True,
            )
            yield {"type": "token", "text": cached.answer}
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            await self._log(tenant, question, result)
            yield self._done_event(result)
            return

        if not sources:
            result = AnswerResult(answer=NO_CONTEXT_ANSWER)
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            await self._log(tenant, question, result)
            yield self._done_event(result)
            return

        messages = build_messages(question, sources, history)
        parts: list[str] = []
        usage = Usage()

        async for delta in self.llm.stream(messages):
            if delta.text:
                parts.append(delta.text)
                yield {"type": "token", "text": delta.text}
            if delta.usage is not None:
                usage = delta.usage

        answer = "".join(parts)
        citations = [c.to_dict() for c in extract_citations(answer, sources)]
        result = AnswerResult(
            answer=answer,
            citations=citations,
            sources=sources,
            usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost_cny=self._cost(usage),
        )
        if use_cache and not history:
            self.cache.put(query_vector, question, answer, citations)

        await self._log(tenant, question, result)
        yield self._done_event(result)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    async def _prepare(
        self,
        question: str,
        tenant: str,
        history: Sequence[dict[str, str]] | None,
        top_k: int | None,
        use_cache: bool,
    ):
        search_query = question
        if history:
            search_query = await self._rewrite(question, history)

        query_vector = await self.embedder.embed_query(search_query)

        if use_cache and self.settings.semantic_cache_enabled and not history:
            cached = self.cache.get(query_vector)
            if cached is not None:
                return search_query, query_vector, [], cached

        sources = await self.retriever.retrieve(
            search_query, tenant, top_k=top_k, query_vector=query_vector
        )
        return search_query, query_vector, sources, None

    async def _rewrite(self, question: str, history: Sequence[dict[str, str]]) -> str:
        """多轮场景下，"那它呢"这类问题需要先补全成独立问题，否则检索不到东西。"""
        try:
            text, _ = await self.llm.complete(
                build_rewrite_messages(question, history),
                temperature=0.0,
                max_tokens=80,
            )
            rewritten = text.strip().strip('"').strip()
            return rewritten or question
        except Exception:  # noqa: BLE001  改写失败不影响主流程
            return question

    def _cost(self, usage: Usage) -> float:
        return round(
            usage.prompt_tokens / 1_000_000 * self.settings.price_input_per_1m
            + usage.completion_tokens / 1_000_000 * self.settings.price_output_per_1m,
            6,
        )

    async def _log(self, tenant: str, question: str, result: AnswerResult) -> None:
        record = QueryLog(
            tenant=tenant,
            question=question,
            answer=result.answer,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_cny=result.cost_cny,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
        )
        await to_thread.run_sync(self.store.log_query, record)

    def _done_event(self, result: AnswerResult) -> dict:
        return {
            "type": "done",
            "answer": result.answer,
            "citations": result.citations,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
            "cost_cny": result.cost_cny,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

    @staticmethod
    def _preview(source: RetrievedChunk, index: int) -> dict:
        text = source.text_for_prompt.replace("\n", " ").strip()
        return {
            "index": index,
            "document_id": source.chunk.document_id,
            "filename": source.chunk.meta.get("filename", "未知文档"),
            "chunk_index": source.chunk.index,
            "snippet": text[:120],
            "score": round(source.score, 4),
        }
