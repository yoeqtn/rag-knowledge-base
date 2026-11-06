"""端到端冒烟测试：入库 -> 检索 -> 问答 -> 引用，不依赖任何外部模型 API。

用法：
    python -m scripts.smoke_test

默认用 memory 存储。验证 Postgres 链路时：
    set STORAGE_BACKEND=postgres
    python -m scripts.smoke_test
"""

from __future__ import annotations

import sys
import time

import anyio

from app.config import Settings
from app.container import build_container
from app.core.llm import FakeChatModel

SAMPLE = """# 差旅报销制度

## 一、适用范围
本制度适用于公司全体正式员工。

## 二、报销流程
员工出差结束后 15 个工作日内，在 OA 系统提交报销申请，附上发票原件扫描件。
直属主管审批通过后转交财务部复核，财务部在 5 个工作日内完成打款。

## 三、住宿标准
一线城市每晚不超过 600 元，新一线城市不超过 450 元，其他城市不超过 350 元。
"""


def wait_for_store(store, attempts: int = 15, delay: float = 1.0) -> None:
    for i in range(attempts):
        health = store.health()
        if health.get("status") == "ok":
            print(f"[ok] storage ready: {health}")
            return
        print(f"[..] waiting for storage ({i + 1}/{attempts}): {health}")
        time.sleep(delay)
    raise SystemExit("存储不可用，请检查 DATABASE_URL 或 docker compose up -d db")


async def main() -> None:
    settings = Settings()
    container = build_container(settings)
    fake = FakeChatModel("根据制度，报销需在出差结束后 15 个工作日内提交申请 [1]。")
    container.llm = fake
    container.qa.llm = fake

    wait_for_store(container.store)

    result = await container.ingestion.ingest("差旅报销制度.md", SAMPLE.encode("utf-8"))
    print(f"[ok] ingest: chunks={result.chunk_count} duplicate={result.duplicate}")

    answer = await container.qa.answer("报销流程是什么？")
    print(f"[ok] answer: {answer.answer}")
    print(f"[ok] citations: {answer.citations}")
    print(f"[ok] latency: {answer.latency_ms}ms  cost: {answer.cost_cny}")

    stats = container.store.stats("default")
    print(f"[ok] stats: {stats}")

    container.store.close()
    print("\n冒烟测试通过 ✅")


if __name__ == "__main__":
    sys.exit(anyio.run(main))
