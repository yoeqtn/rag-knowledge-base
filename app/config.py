"""全局配置。所有可调参数集中在这里，通过 .env 覆盖。"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Knowledge Base"
    api_key: str = "dev-key"

    # ---------- 存储 ----------
    storage_backend: str = "memory"  # memory | postgres
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"

    # ---------- Embedding ----------
    embedding_provider: str = "hash"  # hash | openai
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_batch_size: int = 32

    # ---------- LLM ----------
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ---------- Rerank ----------
    rerank_enabled: bool = False
    rerank_provider: str = "siliconflow"  # siliconflow | jina | dashscope
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""

    # ---------- 切块 ----------
    chunk_strategy: str = "parent_child"  # fixed | recursive | parent_child
    chunk_size: int = 400
    chunk_overlap: int = 60
    parent_chunk_size: int = 1200
    child_chunk_size: int = 300

    # ---------- 检索 ----------
    retrieval_top_k: int = 5
    retrieval_candidates: int = 30
    rrf_k: int = 60
    # 两路召回开关，默认全开。关掉其中一路用于做消融实验。
    retrieval_vector_enabled: bool = True
    retrieval_bm25_enabled: bool = True

    # ---------- 语义缓存 ----------
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95

    # ---------- 成本（元 / 百万 token） ----------
    price_input_per_1m: float = 1.0
    price_output_per_1m: float = 2.0

    max_upload_mb: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
