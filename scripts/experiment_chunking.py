"""切块策略对比实验。

同一批文档、同一批问题，只改切块配置，观察检索效果和上下文长度的变化。
这是简历里"Recall@5 从 X 提升到 Y"这类数字的出处。

用法：
    # 离线、内存存储，几秒钟出结果
    python -m scripts.experiment_chunking --storage memory

    # 用真实 embedding + PostgreSQL（效果才可信）
    python -m scripts.experiment_chunking --storage postgres

    # 只跑其中几个配置
    python -m scripts.experiment_chunking --only parent-300,parent-500

说明：
- 每个配置写入独立的 tenant（exp-xxx），互不干扰
- 重跑前会清空对应的实验 tenant，不影响 default
- 只评测检索层，不调用 LLM，零费用
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anyio
from anyio import to_thread

from app.config import Settings
from app.container import build_container
from app.core.rerank import build_reranker
from app.core.retrieval import Retriever
from app.core.stores import Store
from scripts._eval import evaluate_retrieval, ingest_dir, reset_tenant


@dataclass(slots=True)
class ChunkConfig:
    name: str
    strategy: str
    chunk_size: int
    parent_size: int
    child_size: int


# 实验设计：
#   recursive-200/300/500 与 parent-200/300/500 两两同块长，
#   这样"父子块带来的增益"才和"块变小带来的增益"分得开。
#   fixed-300 作为最朴素的基线。
CONFIGS: list[ChunkConfig] = [
    ChunkConfig("fixed-300", "fixed", 300, 1200, 300),
    ChunkConfig("recursive-200", "recursive", 200, 1200, 200),
    ChunkConfig("recursive-300", "recursive", 300, 1200, 300),
    ChunkConfig("recursive-500", "recursive", 500, 1200, 500),
    ChunkConfig("parent-200", "parent_child", 200, 1200, 200),
    ChunkConfig("parent-300", "parent_child", 300, 1200, 300),
    ChunkConfig("parent-500", "parent_child", 500, 1200, 500),
]


async def evaluate(
    store: Store,
    embedder,
    settings: Settings,
    items: list[dict],
    tenant: str,
    top_k: int,
    query_vectors: list[list[float]],
) -> dict:
    reranker = build_reranker(
        settings.rerank_enabled,
        settings.rerank_provider,
        settings.rerank_model,
        settings.rerank_base_url,
        settings.rerank_api_key,
    )
    retriever = Retriever(store, embedder, reranker, settings)
    metrics = await evaluate_retrieval(retriever, items, tenant, top_k, query_vectors)

    chunks = await to_thread.run_sync(store.all_chunks, tenant, "child")
    return {
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1)
        if chunks
        else 0,
        "recall": metrics.recall,
        "mrr": metrics.mrr,
        "context_hit": metrics.context_hit,
        "keyword_coverage": metrics.keyword_coverage,
        "precision": metrics.precision,
        "avg_context_chars": metrics.avg_context_chars,
        "avg_latency_ms": metrics.avg_latency_ms,
    }


def render_markdown(rows: list[dict], docs_dir: Path, sample_count: int, top_k: int) -> str:
    lines = [
        "# 切块策略对比实验",
        "",
        f"- 文档目录：`{docs_dir}`",
        f"- 问题数：{sample_count}",
        f"- top_k：{top_k}",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 配置 | 策略 | 块数 | 平均块长 | Recall@k | MRR | 上下文命中率 | 检索精度 | 关键词覆盖 | 平均上下文长度 | 检索延迟* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['strategy']} | {row['chunks']} | "
            f"{row['avg_chunk_chars']} | {row['recall']:.2%} | {row['mrr']:.4f} | "
            f"{row['context_hit']:.2%} | {row['precision']:.2%} | "
            f"{row['keyword_coverage']:.2%} | "
            f"{row['avg_context_chars']} | {row['avg_latency_ms']}ms |"
        )

    best = (
        max(
            rows,
            key=lambda r: (
                r["context_hit"],
                r["precision"],
                r["recall"],
                r["mrr"],
                -r["avg_context_chars"],  # 同等效果下，上下文越短越省钱
            ),
        )
        if rows
        else None
    )
    if best:
        lines += [
            "",
            f"**综合最优：`{best['name']}`**（命中率 {best['context_hit']:.2%}，"
            f"精度 {best['precision']:.2%}，MRR {best['mrr']:.4f}，"
            f"平均上下文 {best['avg_context_chars']} 字）",
        ]

    recalls = {round(row["recall"], 4) for row in rows}
    context_hits = {round(row["context_hit"], 4) for row in rows}
    if len(recalls) == 1 and len(context_hits) == 1:
        lines += [
            "",
            "> ⚠️ **所有配置结果完全相同，说明这套题/这批文档没有区分度。**",
            "> 常见原因：文档太少、问题直接抄了原文、top_k 大到能覆盖所有块。",
            "> 这样的数字不能写进简历。请扩充文档（至少 30~50 篇，加入同类干扰文档），",
            "> 并把问题改写成不含原文关键词的口语化表达，再重跑。",
        ]
    else:
        lines += [
            "",
            "> 注意：结论只对这批文档和这批问题成立。换语料要重跑。",
        ]

    lines += [
        "",
        "\\* 检索延迟只统计向量检索 + BM25 + RRF，不含 query embedding，"
        "因为 embedding 成本对所有配置相同，算进去会掩盖切块本身的差异。",
        "",
        "---",
        "",
        "## 指标说明（怎么读这张表）",
        "",
        f"本次共 {sample_count} 道题，top_k={top_k}，即每题检索返回 {top_k} 个块。",
        "",
        "| 指标 | 大白话 | 怎么算 | 方向 |",
        "|---|---|---|---|",
        "| 块数 | 检索库里一共有几个块 | 入库后统计 | 不是越大越好，越多越占内存 |",
        "| 平均块长 | 每个块平均多少个字 | 总字数 ÷ 块数 | 不是越大越好 |",
        "| Recall@k | 该找的文档，找到了吗 | 命中题数 ÷ 总题数 | 越高越好 |",
        "| MRR | 找到的文档排在第几位 | 平均(1 ÷ 排名)，排第 1 得 1 分 | 越高越好，1.0 满分 |",
        "| 上下文命中率 | 答案需要的词，是否全部进了上下文 | 全命中的题数 ÷ 总题数 | 越高越好 |",
        "| 检索精度 | 返回的块里，几个是真正有用的 | 平均(含答案的块数 ÷ 返回块数) | 越高越好 |",
        "| 关键词覆盖 | 答案关键词命中了几成 | 平均(命中词数 ÷ 总词数) | 越高越好 |",
        "| 平均上下文长度 | 每次问答题库塞给模型多少字 | 总字符数 ÷ 总题数 | 越低越省钱 |",
        "| 检索延迟 | 查一次要多久 | 平均毫秒 | 越低越好 |",
        "",
        "### 拿一行数据翻译一下",
        "",
        "以 `parent-200` 为例：",
        "",
        "- **块数 49**：11 篇文档被切成 49 个可检索子块",
        "- **平均块长 173.3**：每个子块平均 173 个字",
        "- **Recall@3 100%**：17 道题，每道题的答案所在文档都进了前 3 名",
        "- **MRR 1.0000**：而且每道题的正确文档都排在第 1 位",
        "- **上下文命中率 100%**：17 道题需要的答案关键词，全部出现在喂给模型的上下文里",
        "- **检索精度 74.51%**：每题返回 3 个块，平均 2.24 个是真正含答案的（0.7451 × 3）",
        "- **平均上下文长度 953.6 字**：每题大约塞给模型 954 个字",
        "- **检索延迟 63ms**：查一次大约 63 毫秒",
        "",
        "> 注意：Recall / 精度 / 命中率衡量的是检索的好坏，"
        "块数 / 块长 / 上下文长度 / 延迟衡量的是代价。"
        "真实选型是两边的权衡，不存在所有指标都最优的配置。",
    ]
    return "\n".join(lines)


async def run(args) -> int:
    base = Settings()
    if args.storage:
        base = base.model_copy(update={"storage_backend": args.storage})
    # 比较切块策略时必须关掉重排，否则重排会把切块的差异抹平
    if args.no_rerank:
        base = base.model_copy(update={"rerank_enabled": False})

    docs_dir = Path(args.docs)
    items = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        print("[!] 评测集为空")
        return 1

    selected = CONFIGS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        selected = [config for config in CONFIGS if config.name in wanted]
        if not selected:
            print(f"[!] 没有匹配的配置: {args.only}")
            return 1

    container = build_container(base)  # 复用一个 store 和 embedder，避免重复加载
    store, embedder = container.store, container.embedder

    print(f"[i] storage={base.storage_backend}  embedder={base.embedding_provider} "
          f"dim={embedder.dim}  documents={docs_dir}  questions={len(items)}")

    # 所有 query 向量只算一次，各配置复用。
    # 这样延迟差异只来自检索本身，而不是"谁先跑谁付 embedding 的钱"。
    query_vectors = [await embedder.embed_query(item["question"]) for item in items]

    rows: list[dict] = []
    for config in selected:
        tenant = f"exp-{config.name}"
        settings = base.model_copy(
            update={
                "chunk_strategy": config.strategy,
                "chunk_size": config.chunk_size,
                "parent_chunk_size": config.parent_size,
                "child_chunk_size": config.child_size,
            }
        )

        await reset_tenant(store, tenant)
        started = time.perf_counter()
        await ingest_dir(store, embedder, settings, docs_dir, tenant)
        ingest_seconds = time.perf_counter() - started

        metrics = await evaluate(
            store, embedder, settings, items, tenant, args.top_k, query_vectors
        )
        rows.append(
            {
                "name": config.name,
                "strategy": config.strategy,
                **metrics,
                "ingest_s": round(ingest_seconds, 2),
            }
        )
        print(
            f"[{config.name:<14}] chunks={metrics['chunks']:<4} "
            f"recall={metrics['recall']:.2%}  mrr={metrics['mrr']:.4f}  "
            f"ctx_hit={metrics['context_hit']:.2%}  "
            f"prec={metrics['precision']:.2%}  "
            f"ctx={metrics['avg_context_chars']}chars  "
            f"latency={metrics['avg_latency_ms']}ms  ({ingest_seconds:.1f}s)"
        )

    print("\n" + render_markdown(rows, docs_dir, len(items), args.top_k))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(rows, docs_dir, len(items), args.top_k), "utf-8")
        print(f"\n[done] 已写入 {output}")

    container.store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="切块策略对比实验")
    parser.add_argument("--docs", default="data/sample")
    parser.add_argument("--dataset", default="eval/dataset.example.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--storage", default=None, choices=["memory", "postgres"])
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="强制关闭重排，保证比较的只是切块策略",
    )
    parser.add_argument("--only", default=None, help="逗号分隔的配置名，例如 parent-300,parent-500")
    parser.add_argument(
        "--output",
        default="results/chunking-comparison.md",
        help="结果写入的 markdown 路径",
    )
    return anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
