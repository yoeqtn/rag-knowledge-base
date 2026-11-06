"""检索/问答效果评测。

用法：
    # 只看检索（不需要 LLM，零成本）
    python -m scripts.evaluate --offline

    # 完整评测（需要 .env 配好 LLM，会产生费用）
    python -m scripts.evaluate

指标：
- source_hit@k : 期望文档是否出现在 top-k 检索结果里
- MRR          : 期望文档在检索结果中的排名倒数，越接近 1 越好
- keyword      : 答案命中了多少条期望关键词（近似衡量答案正确性）
- latency/cost : 平均延迟与总花费
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anyio

from app.config import Settings
from app.container import build_container
from app.core.parsing import SUPPORTED_EXTENSIONS


async def ingest_dir(container, directory: Path, tenant: str) -> None:
    files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    for path in files:
        result = await container.ingestion.ingest(path.name, path.read_bytes(), tenant)
        print(f"[ingest] {path.name} chunks={result.chunk_count} duplicate={result.duplicate}")


async def run(args) -> int:
    settings = Settings()
    container = build_container(settings)

    if args.docs:
        await ingest_dir(container, args.docs, args.tenant)

    items = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        print("[!] 数据集为空")
        return 1

    if not settings.llm_api_key and not args.offline:
        print("[!] 未配置 LLM_API_KEY，自动切换为离线评测")
    offline = args.offline or not settings.llm_api_key

    print(
        f"\n数据集: {args.dataset}  样本数: {len(items)}  top_k={args.top_k}  "
        f"offline={offline}\n"
    )
    print(f"{'#':<3}{'source@k':<10}{'rank':<6}{'keyword':<10}{'latency':<10}question")
    print("-" * 90)

    hits = 0
    mrr_total = 0.0
    coverage_values: list[float] = []
    latencies: list[int] = []
    total_cost = 0.0

    for index, item in enumerate(items, start=1):
        question = item["question"]
        expected = item.get("expected_source")

        sources = await container.retriever.retrieve(question, args.tenant, top_k=args.top_k)
        filenames = [source.chunk.meta.get("filename") for source in sources]
        rank = filenames.index(expected) + 1 if expected in filenames else None
        if rank:
            hits += 1
            mrr_total += 1.0 / rank

        keyword_text = "-"
        latency = 0
        if not offline:
            result = await container.qa.answer(question, args.tenant, use_cache=False)
            keywords = item.get("expected_keywords") or []
            if keywords:
                matched = sum(1 for keyword in keywords if keyword in result.answer)
                coverage_values.append(matched / len(keywords))
                keyword_text = f"{matched}/{len(keywords)}"
            latency = result.latency_ms
            latencies.append(latency)
            total_cost += result.cost_cny

        print(
            f"{index:<3}{'✓' if rank else '✗':<10}"
            f"{rank or '-':<6}{keyword_text:<10}{latency or '-':<10}{question}"
        )

    total = len(items)
    print("-" * 90)
    print(f"source_hit@{args.top_k}: {hits}/{total} = {hits / total:.2%}")
    print(f"MRR:            {mrr_total / total:.4f}")
    if coverage_values:
        print(f"keyword 命中率: {sum(coverage_values) / len(coverage_values):.2%}")
    if latencies:
        print(f"平均延迟:       {sum(latencies) / len(latencies):.0f}ms")
        print(f"总花费:         ¥{total_cost:.6f}")

    container.store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 效果评测")
    parser.add_argument("--dataset", default="eval/dataset.example.jsonl")
    parser.add_argument("--docs", type=Path, default=None, help="评测前先导入的文档目录")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--offline", action="store_true", help="只评测检索，不调用 LLM")
    return anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
