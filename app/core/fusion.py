"""多路召回融合。

用 RRF（Reciprocal Rank Fusion）而不是直接分数相加，原因：
- 向量相似度（0~1）和 BM25 分数（0~几十）量纲完全不同，直接加权很难调
- RRF 只看排名，天然对两路分数的尺度不敏感，工业界常用且极其稳定

score(d) = Σ 1 / (k + rank_i(d))，k 一般取 60。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """输入多路有序 id 列表，输出融合后的 (id, score)，按分数降序。"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 数量必须与 ranked_lists 一致")

    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
