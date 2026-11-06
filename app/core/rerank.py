"""重排序（可选）。

不同厂商的 rerank 接口格式不一样，这里做了一层适配：

- siliconflow / jina：POST /rerank，扁平的 query/documents/top_n
- 阿里云百炼 DashScope：POST /services/rerank/text-rerank/text-rerank，
  参数放在 input/parameters 里，结果在 output.results 里

"怎么发请求、怎么读响应"抽成纯函数，既能单测，也方便再加一家。
"""

from __future__ import annotations

from typing import Protocol, Sequence

import httpx

PROVIDERS = ("siliconflow", "jina", "dashscope")


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        """返回 (原文档下标, 相关性分数)，已按分数降序。"""
        ...


class NoopReranker:
    async def rerank(
        self, query: str, documents: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]


# --------------------------------------------------------------------------- #
# 纯函数：请求 / 响应格式
# --------------------------------------------------------------------------- #
def build_rerank_request(
    provider: str, model: str, query: str, documents: Sequence[str], top_k: int
) -> dict:
    docs = list(documents)
    top_n = min(top_k, len(docs))

    if provider == "dashscope":
        return {
            "model": model,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": False, "top_n": top_n},
        }

    # siliconflow / jina
    return {
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": top_n,
    }


def parse_rerank_response(payload: dict) -> list[tuple[int, float]]:
    """兼容三种常见结构，按分数降序返回。"""
    results = payload.get("results") or payload.get("data")
    if results is None:
        results = (payload.get("output") or {}).get("results") or []

    pairs: list[tuple[int, float]] = []
    for item in results:
        score = item.get("relevance_score", item.get("score", 0.0))
        pairs.append((int(item["index"]), float(score)))

    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# 实现
# --------------------------------------------------------------------------- #
class ApiReranker:
    """兼容 siliconflow / jina / 阿里云百炼 的 HTTP 重排客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "siliconflow",
        timeout: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("RERANK_ENABLED=true 时必须配置 RERANK_API_KEY")
        self.provider = provider.lower()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"未知 RERANK_PROVIDER: {provider}，可选 {', '.join(PROVIDERS)}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _endpoint(self) -> str:
        base = self.base_url
        if self.provider == "dashscope":
            # 已经给了完整路径就原样用
            if "services/rerank" in base:
                return base
            return f"{base}/services/rerank/text-rerank/text-rerank"
        if base.endswith("/rerank"):
            return base
        return f"{base}/rerank"

    async def rerank(
        self, query: str, documents: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]:
        payload = build_rerank_request(
            self.provider, self.model, query, documents, top_k
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._endpoint(),
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return parse_rerank_response(data)[:top_k]


def build_reranker(
    enabled: bool,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Reranker:
    if not enabled:
        return NoopReranker()
    return ApiReranker(
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
    )
