"""召回方式对比实验：纯向量 vs 纯 BM25 vs 混合召回。

固定切块配置，只改检索链路，回答一个问题：
"两路召回 + RRF 融合，到底比单路强多少？"

用法：
    python -m scripts.experiment_retrieval
    python -m scripts.experiment_retrieval --dataset eval/dataset.hard.jsonl --top-k 3
    python -m scripts.experiment_retrieval --storage memory     # 离线，用假向量
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anyio

from app.config import Settings
from app.container import build_container
from app.core.rerank import NoopReranker, build_reranker
from app.core.retrieval import Retriever
from scripts._eval import evaluate_retrieval, ingest_dir, reset_tenant


@dataclass(slots=True)
class Arm:
    name: str
    vector: bool
    bm25: bool
    rerank: bool
    note: str


ARMS: list[Arm] = [
    Arm("vector-only", True, False, False, "只走向量召回"),
    Arm("bm25-only", False, True, False, "只走关键词召回"),
    Arm("hybrid", True, True, False, "两路召回 + RRF 融合"),
    Arm("hybrid+rerank", True, True, True, "两路召回 + RRF + 重排"),
]


async def run(args) -> int:
    base = Settings()
    if args.storage:
        base = base.model_copy(update={"storage_backend": args.storage})

    docs_dir = Path(args.docs)
    items = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        print("[!] 评测集为空")
        return 1

    selected = ARMS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        selected = [arm for arm in ARMS if arm.name in wanted]
        if not selected:
            print(f"[!] 没有匹配的召回方式: {args.only}")
            return 1

    container = build_container(base)
    store, embedder = container.store, container.embedder
    tenant = "exp-retrieval"

    print(
        f"[i] storage={base.storage_backend}  embedder={base.embedding_provider}  "
        f"chunk={base.chunk_strategy}/{base.child_chunk_size}  "
        f"documents={docs_dir}  questions={len(items)}"
    )

    await reset_tenant(store, tenant)
    started = time.perf_counter()
    chunks = await ingest_dir(store, embedder, base, docs_dir, tenant)
    ingest_seconds = time.perf_counter() - started
    print(f"[i] ingested {chunks} chunks in {ingest_seconds:.1f}s")

    # 所有实验组共用同一批 query 向量，避免"谁先跑谁付 embedding 的钱"
    query_vectors = [await embedder.embed_query(item["question"]) for item in items]

    rows: list[dict] = []
    details_by_arm: dict[str, list[dict]] = {}

    for arm in selected:
        if arm.rerank:
            if not base.rerank_enabled:
                print("[!] 跳过 hybrid+rerank：.env 里 RERANK_ENABLED=false")
                continue
            try:
                reranker = build_reranker(
                    True,
                    base.rerank_provider,
                    base.rerank_model,
                    base.rerank_base_url,
                    base.rerank_api_key,
                )
            except ValueError as exc:
                print(f"[!] 跳过 hybrid+rerank：{exc}")
                continue
        else:
            reranker = NoopReranker()

        settings = base.model_copy(
            update={
                "retrieval_vector_enabled": arm.vector,
                "retrieval_bm25_enabled": arm.bm25,
                "rerank_enabled": arm.rerank,
            }
        )
        retriever = Retriever(store, embedder, reranker, settings)
        metrics = await evaluate_retrieval(
            retriever, items, tenant, args.top_k, query_vectors
        )
        rows.append(
            {
                "name": arm.name,
                "note": arm.note,
                "recall": metrics.recall,
                "mrr": metrics.mrr,
                "context_hit": metrics.context_hit,
                "precision": metrics.precision,
                "avg_context_chars": metrics.avg_context_chars,
                "avg_latency_ms": metrics.avg_latency_ms,
            }
        )
        details_by_arm[arm.name] = metrics.details
        print(
            f"[{arm.name:<14}] recall={metrics.recall:.2%}  mrr={metrics.mrr:.4f}  "
            f"ctx_hit={metrics.context_hit:.2%}  prec={metrics.precision:.2%}  "
            f"ctx={metrics.avg_context_chars}  latency={metrics.avg_latency_ms}ms"
        )

    report = render_markdown(
        rows,
        details_by_arm,
        len(items),
        args.top_k,
        base,
        docs_dir,
        show_details=args.details,
    )
    print("\n" + report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, "utf-8")
        print(f"\n[done] 已写入 {output}")

    container.store.close()
    return 0


def _describe(state: tuple) -> str:
    rank, hit = state
    if rank is None:
        return "未召回"
    return f"第{rank}名{' ✅' if hit else ' ❌'}"


def render_markdown(
    rows: list[dict],
    details_by_arm: dict[str, list[dict]],
    sample_count: int,
    top_k: int,
    settings: Settings,
    docs_dir: Path,
    show_details: bool = False,
) -> str:
    lines = [
        "# 召回方式对比实验",
        "",
        f"- 文档目录：`{docs_dir}`",
        f"- 问题数：{sample_count}",
        f"- top_k：{top_k}",
        f"- 切块配置：{settings.chunk_strategy} / 子块 {settings.child_chunk_size} 字",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 召回方式 | 说明 | Recall@k | MRR | 上下文命中率 | 检索精度 | 平均上下文 | 检索延迟* |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['note']} | {row['recall']:.2%} | "
            f"{row['mrr']:.4f} | {row['context_hit']:.2%} | {row['precision']:.2%} | "
            f"{row['avg_context_chars']} | {row['avg_latency_ms']}ms |"
        )

    by_name = {row["name"]: row for row in rows}
    if "vector-only" in by_name and "hybrid" in by_name:
        vector, hybrid = by_name["vector-only"], by_name["hybrid"]
        lines += [
            "",
            "**混合召回 vs 纯向量**：",
            f"上下文命中率 {vector['context_hit']:.2%} → {hybrid['context_hit']:.2%}，"
            f"检索精度 {vector['precision']:.2%} → {hybrid['precision']:.2%}，"
            f"MRR {vector['mrr']:.4f} → {hybrid['mrr']:.4f}。",
        ]
    if "bm25-only" in by_name:
        bm25 = by_name["bm25-only"]
        lines.append(
            f"纯 BM25 作为参照：命中率 {bm25['context_hit']:.2%}，"
            f"精度 {bm25['precision']:.2%}，MRR {bm25['mrr']:.4f}。"
        )

    lines += [
        "",
        f"> ⚠️ 样本量提醒：{sample_count} 道题里，1 道题的差异 = "
        f"{100 / sample_count:.1f} 个百分点。小于这个数的差距很可能只是单题波动，不能当结论。",
    ]

    # 找出不同召回方式结果不一致的题——这些题才能说明两路的差异。
    # 注意要比"排名 + 是否命中"，只比命中会漏掉"排第 1 还是排第 3"的区别。
    if len(details_by_arm) >= 2:
        names = list(details_by_arm)
        mismatches = []
        for index in range(len(details_by_arm[names[0]])):
            states = {
                name: (
                    details_by_arm[name][index]["rank"],
                    bool(details_by_arm[name][index]["context_hit"]),
                )
                for name in names
            }
            if len(set(states.values())) > 1:
                mismatches.append(
                    {
                        "question": details_by_arm[names[0]][index]["question"],
                        "states": states,
                    }
                )

        if mismatches:
            lines += [
                "",
                "## 分歧题（这些题才能说明两路的差别）",
                "",
                f"| 问题 | {' | '.join(names)} |",
                "|---|" + "---|" * len(names),
            ]
            for item in mismatches:
                cells = " | ".join(_describe(item["states"][name]) for name in names)
                lines.append(f"| {item['question']} | {cells} |")
            lines += [
                "",
                "> 格式为「排名 + 答案是否进上下文」；未召回表示该文档没进 top-k。",
            ]
        else:
            lines += [
                "",
                "> 所有题在三组召回下的排名完全一致，这套题区分不出两路的能力差异。",
            ]

    if show_details and details_by_arm:
        names = list(details_by_arm)
        differing = []
        for index in range(len(details_by_arm[names[0]])):
            values = [
                details_by_arm[name][index]["precision"] for name in names
            ]
            if len({round(v, 4) if v is not None else None for v in values}) > 1:
                differing.append(
                    {
                        "question": details_by_arm[names[0]][index]["question"],
                        "values": values,
                    }
                )

        if differing:
            lines += [
                "",
                "## 各题检索精度（有用的块数 / 返回的块数）",
                "",
                f"| 问题 | {' | '.join(names)} |",
                "|---|" + "---|" * len(names),
            ]
            for item in differing:
                cells = " | ".join(
                    "-" if value is None else f"{value:.2f}" for value in item["values"]
                )
                lines.append(f"| {item['question']} | {cells} |")
            lines.append("")
            lines.append("> 0.33 表示 3 个块里只有 1 个含答案，1.00 表示 3 个全中。")

    lines += [
        "",
        "\\* 检索延迟包含向量检索 + BM25 + RRF，启用重排时也包含重排请求；不含 query embedding。",
        "",
        "---",
        "",
        "## 指标说明",
        "",
        "| 指标 | 大白话 | 方向 |",
        "|---|---|---|",
        "| Recall@k | 该找的文档，找到了吗 | 越高越好 |",
        "| MRR | 找到的文档排第几，第 1 名得 1 分 | 越高越好，1.0 满分 |",
        "| 上下文命中率 | 答案需要的词是否全部进了上下文 | 越高越好 |",
        "| 检索精度 | 返回的块里，几个是真正有用的 | 越高越好 |",
        "| 平均上下文 | 每次塞给模型多少字 | 越低越省钱 |",
        "| 检索延迟 | 查一次要多久 | 越低越好 |",
        "",
        "> 注意：这批语料只有十几篇文档，任何结论都应理解为方向性的。",
        "> 写进简历前请扩到 50 篇以上、100 条问题以上重跑。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="召回方式对比实验")
    parser.add_argument("--docs", default="data/sample")
    parser.add_argument("--dataset", default="eval/dataset.hard.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--storage", default=None, choices=["memory", "postgres"])
    parser.add_argument("--only", default=None, help="逗号分隔，例如 vector-only,hybrid")
    parser.add_argument(
        "--details",
        action="store_true",
        help="在报告里列出各题检索精度，便于分析差异来自哪几道题",
    )
    parser.add_argument("--output", default="results/retrieval-comparison.md")
    return anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
