"""向量化。

对外只暴露 Embedder 协议，具体实现可替换：
- HashEmbedder   ：本地确定性向量，仅用于测试和离线跑通链路
- OpenAIEmbedder ：任何 OpenAI 兼容的 /embeddings 接口（硅基流动、通义、智谱…）
- CachedEmbedder ：带 LRU 缓存的装饰器，避免对同一段文本重复付费
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import OrderedDict
from typing import Protocol, Sequence

import numpy as np

from app.config import Settings


class Embedder(Protocol):
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def l2_normalize(vector: Sequence[float]) -> list[float]:
    """归一化后，余弦相似度 = 点积，检索时省一次除法。"""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [0.0] * len(arr)
    return (arr / norm).tolist()


# --------------------------------------------------------------------------- #
# Hash 实现：确定性、零依赖、零成本，但语义能力很弱
# --------------------------------------------------------------------------- #
class HashEmbedder:
    """signed feature hashing：把词映射到固定维度，用于测试与离线演示。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        import jieba

        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        return l2_normalize(vec)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# --------------------------------------------------------------------------- #
# OpenAI 兼容实现
# --------------------------------------------------------------------------- #
class OpenAIEmbedder:
    def __init__(
        self,
        model: str,
        dim: int,
        base_url: str,
        api_key: str,
        batch_size: int = 32,
        concurrency: int = 4,
    ) -> None:
        from openai import AsyncOpenAI

        if not api_key:
            raise ValueError(
                "EMBEDDING_API_KEY 为空。请到 .env 填写，或改用 EMBEDDING_PROVIDER=hash"
            )
        self.dim = dim
        self.model = model
        self.batch_size = batch_size
        self._concurrency = concurrency
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model,
            input=batch,
            encoding_format="float",
        )
        # API 不保证按顺序返回，按 index 排序
        items = sorted(response.data, key=lambda x: x.index)
        vectors = [l2_normalize(item.embedding) for item in items]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"向量维度不匹配：模型返回 {len(vectors[0])}，配置中是 {self.dim}。"
                f"请修改 .env 里的 EMBEDDING_DIM"
            )
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batches = [
            list(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        semaphore = asyncio.Semaphore(self._concurrency)

        async def run(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch(batch)

        results = await asyncio.gather(*(run(b) for b in batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]


# --------------------------------------------------------------------------- #
# 缓存装饰器
# --------------------------------------------------------------------------- #
class CachedEmbedder:
    """按文本哈希缓存向量。重传同一份文档时能省掉大头的 embedding 费用。"""

    def __init__(self, inner: Embedder, max_entries: int = 5000) -> None:
        self.inner = inner
        self.dim = inner.dim
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get(self, text: str) -> list[float] | None:
        key = self._key(text)
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
            self.hits += 1
        return value

    def _put(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        missing: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = self._key(text)
            if self._get(text) is None and key not in seen:
                seen.add(key)
                missing.append(text)

        if missing:
            self.misses += len(missing)
            vectors = await self.inner.embed_documents(missing)
            for text, vector in zip(missing, vectors):
                self._put(text, vector)

        return [self._get(text) or [0.0] * self.dim for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        cached = self._get(text)
        if cached is not None:
            return cached
        vector = await self.inner.embed_query(text)
        self._put(text, vector)
        return vector


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "hash":
        # 维度必须和 EMBEDDING_DIM 一致，否则和 pgvector 的列宽对不上
        return CachedEmbedder(HashEmbedder(dim=settings.embedding_dim))
    if provider == "openai":
        return CachedEmbedder(
            OpenAIEmbedder(
                model=settings.embedding_model,
                dim=settings.embedding_dim,
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                batch_size=settings.embedding_batch_size,
            )
        )
    raise ValueError(f"未知 EMBEDDING_PROVIDER: {settings.embedding_provider}")
