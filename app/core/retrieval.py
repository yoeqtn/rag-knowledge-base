"""混合检索：向量召回 + BM25 召回 -> RRF 融合 -> 可选重排 -> 父块回溯。"""

from __future__ import annotations

from anyio import to_thread

from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.embedding import Embedder
from app.core.fusion import reciprocal_rank_fusion
from app.core.rerank import Reranker
from app.core.stores import Store
from app.core.types import RetrievedChunk


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings
        self._bm25_cache: dict[tuple[str, int], BM25Index] = {}

    def _bm25_index(self, tenant: str) -> BM25Index:
        """按 (tenant, store 版本号) 缓存倒排索引；文档一变缓存自动失效。"""
        key = (tenant, self.store.version())
        index = self._bm25_cache.get(key)
        if index is None:
            chunks = self.store.all_chunks(tenant, role="child")
            index = BM25Index().build([(c.id, c.text) for c in chunks])
            self._bm25_cache = {key: index}  # 只留最新一份，避免内存膨胀
        return index

    async def retrieve(
        self,
        query: str,
        tenant: str,
        top_k: int | None = None,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        candidates = max(self.settings.retrieval_candidates, top_k * 3)

        vector_hits: list[tuple] = []
        bm25_hits: list[tuple] = []

        # 1) 向量召回
        if self.settings.retrieval_vector_enabled:
            if query_vector is None:
                query_vector = await self.embedder.embed_query(query)
            vector_hits = await to_thread.run_sync(
                self.store.vector_search, query_vector, tenant, candidates
            )

        # 2) 关键词召回
        if self.settings.retrieval_bm25_enabled:
            bm25_hits = await to_thread.run_sync(
                lambda: self._bm25_index(tenant).search(query, candidates)
            )

        if not vector_hits and not bm25_hits:
            return []

        # 3) RRF 融合。两路都命中的块会拿到双份分数，自然排到前面
        vector_ids = [chunk.id for chunk, _ in vector_hits]
        bm25_ids = [chunk_id for chunk_id, _ in bm25_hits]
        ranked_lists = [ids for ids in (vector_ids, bm25_ids) if ids]
        fused = reciprocal_rank_fusion(ranked_lists, k=self.settings.rrf_k)

        by_id = {chunk.id: chunk for chunk, _ in vector_hits}
        missing = [cid for cid, _ in fused if cid not in by_id]
        if missing:
            for chunk in await to_thread.run_sync(
                self.store.get_chunks_by_ids, missing, tenant
            ):
                by_id[chunk.id] = chunk

        matched_by: dict[str, list[str]] = {}
        for cid in vector_ids:
            matched_by.setdefault(cid, []).append("vector")
        for cid in bm25_ids:
            matched_by.setdefault(cid, []).append("bm25")

        results = [
            RetrievedChunk(
                chunk=by_id[cid],
                score=score,
                matched_by=tuple(matched_by.get(cid, [])),
            )
            for cid, score in fused
            if cid in by_id
        ]

        # 4) 可选重排：只对融合后的前 20 条做，控制延迟
        if self.settings.rerank_enabled and results:
            head = results[:20]
            pairs = await self.reranker.rerank(
                query, [item.text_for_prompt for item in head], top_k=min(top_k, len(head))
            )
            results = [head[i] for i, _ in pairs] + results[len(head) :]
            for item, (_, score) in zip(results, pairs):
                item.score = score
                item.matched_by = (*item.matched_by, "rerank")

        results = results[:top_k]

        # 5) 父块回溯：子块负责"查得准"，父块负责"给 LLM 足够上下文"
        results = await self._expand_parents(results, tenant)
        return results

    async def _expand_parents(
        self, results: list[RetrievedChunk], tenant: str
    ) -> list[RetrievedChunk]:
        parent_ids = [r.chunk.parent_id for r in results if r.chunk.parent_id]
        if not parent_ids:
            return results

        parents = await to_thread.run_sync(
            self.store.get_chunks_by_ids, list(dict.fromkeys(parent_ids)), tenant
        )
        parent_text = {p.id: p.text for p in parents}

        expanded: list[RetrievedChunk] = []
        seen: set[str] = set()
        for item in results:
            parent_id = item.chunk.parent_id
            if parent_id:
                if parent_id in seen:  # 同一个父块下的多个子块，只保留一次
                    continue
                seen.add(parent_id)
                item.context = parent_text.get(parent_id)
            expanded.append(item)
        return expanded
