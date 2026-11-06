"""依赖装配。

整个应用只有一个地方 new 具体实现，其余代码只依赖协议。
测试时把 FakeChatModel / MemoryStore 换进来即可。
"""
from __future__ import annotations
from dataclasses import dataclass
from app.config import Settings, get_settings
from app.core.cache import SemanticCache
from app.core.embedding import Embedder, build_embedder
from app.core.llm import ChatModel, build_chat_model
from app.core.rerank import build_reranker
from app.core.retrieval import Retriever
from app.core.stores import Store, build_store
from app.services.ingest import IngestionService
from app.services.qa import QAService

@dataclass
class Container:
    settings: Settings
    store: Store
    embedder: Embedder
    llm: ChatModel
    retriever: Retriever
    cache: SemanticCache
    ingestion: IngestionService
    qa: QAService

def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()

    store = build_store(settings)
    embedder = build_embedder(settings)
    llm = build_chat_model(settings)
    reranker = build_reranker(
        settings.rerank_enabled,
        settings.rerank_provider,
        settings.rerank_model,
        settings.rerank_base_url,
        settings.rerank_api_key,
    )
    retriever = Retriever(store=store, embedder=embedder, reranker=reranker, settings=settings)
    cache = SemanticCache(
        threshold=settings.semantic_cache_threshold,
        max_entries=512,
    )

    return Container(
        settings=settings,
        store=store,
        embedder=embedder,
        llm=llm,
        retriever=retriever,
        cache=cache,
        ingestion=IngestionService(store, embedder, settings),
        qa=QAService(
            store=store,
            retriever=retriever,
            llm=llm,
            embedder=embedder,
            cache=cache,
            settings=settings,
        ),
    )
