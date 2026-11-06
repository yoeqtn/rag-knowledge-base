"""检查 rerank 配置是否可用。

用途：换服务商后，先确认 key / 地址 / 模型名 三者对得上，再去跑实验。
不会打印 api key。

用法：
    python -m scripts.check_rerank
    python -m scripts.check_rerank --url https://dashscope.aliyuncs.com/api/v1
"""

from __future__ import annotations

import argparse
import sys

import anyio
import httpx

from app.config import Settings
from app.core.rerank import ApiReranker

DOCUMENTS = [
    "一线城市住宿每晚不超过 600 元，新一线城市不超过 450 元。",
    "密码长度不少于 12 位，每 90 天更换一次。",
    "入职满一年享受 5 天年假。",
]
QUERY = "成都出差住宿最多报多少"


async def probe(provider: str, base_url: str, model: str, api_key: str) -> tuple[bool, str]:
    try:
        reranker = ApiReranker(model, base_url, api_key, provider)
    except ValueError as exc:
        return False, str(exc)

    endpoint = reranker._endpoint()
    try:
        pairs = await reranker.rerank(QUERY, DOCUMENTS, top_k=2)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.replace("\n", " ")[:200]
        return False, f"{endpoint} -> HTTP {exc.response.status_code}: {body}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{endpoint} -> {type(exc).__name__}: {exc}"

    if not pairs:
        return False, f"{endpoint} -> HTTP 200 但没解析出结果（响应格式可能变了）"

    top_index, top_score = pairs[0]
    preview = DOCUMENTS[top_index][:20]
    return True, f"{endpoint} -> ok  最高分 {top_score:.4f}  «{preview}…»"


async def run(args) -> int:
    settings = Settings()
    base_url = args.url or settings.rerank_base_url
    provider = args.provider or settings.rerank_provider
    model = args.model or settings.rerank_model

    print("[i] 当前 .env 配置：")
    print(f"    RERANK_ENABLED  = {settings.rerank_enabled}")
    print(f"    RERANK_PROVIDER = {settings.rerank_provider}")
    print(f"    RERANK_MODEL    = {settings.rerank_model}")
    print(f"    RERANK_BASE_URL = {settings.rerank_base_url}")
    print(f"    RERANK_API_KEY  = {'SET' if settings.rerank_api_key else 'EMPTY'}")
    print()

    if not settings.rerank_api_key:
        print("[x] RERANK_API_KEY 为空，先填 key")
        return 1

    print(f"[i] 测试 provider={provider}  url={base_url}\n")
    ok, message = await probe(provider, base_url, model, settings.rerank_api_key)
    print("[✓] " + message if ok else "[x] " + message)

    if not ok and provider == "siliconflow" and "dashscope" in base_url:
        alt = base_url.replace("/compatible-mode/v1", "/api/v1")
        print(f"\n[i] 百炼的 rerank 不在 compatible-mode 下，改用 native 地址重试：{alt}")
        ok, message = await probe("dashscope", alt, model, settings.rerank_api_key)
        print("[✓] " + message if ok else "[x] " + message)

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 rerank 配置")
    parser.add_argument("--provider", default=None, choices=["siliconflow", "jina", "dashscope"])
    parser.add_argument("--url", default=None)
    parser.add_argument("--model", default=None)
    return anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
