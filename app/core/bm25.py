"""BM25 关键词检索（纯 Python 实现，中文用 jieba 分词）。

为什么向量检索之外还要 BM25：
- 向量擅长语义相近，但对"型号、编号、专有名词"这类精确匹配很弱
- BM25 擅长精确词命中，但对同义改写无能为力
两者互补，用 RRF 融合，召回率通常能明显高于单路。
"""

from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter, defaultdict


def _is_noise(token: str) -> bool:
    if not token.strip():
        return True
    return all(
        unicodedata.category(ch)[0] in ("P", "Z", "S") or ch in string.punctuation
        for ch in token
    )


def tokenize(text: str) -> list[str]:
    import jieba

    return [
        token
        for token in (t.strip().lower() for t in jieba.lcut(text))
        if token and not _is_noise(token)
    ]


class BM25Index:
    """标准 Okapi BM25。语料规模在十万块以内时，纯 Python 完全够用。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.avgdl: float = 0.0

    def build(self, documents: list[tuple[str, str]]) -> "BM25Index":
        """documents: [(chunk_id, text), ...]"""
        self.doc_ids = []
        self.doc_len = []
        self.postings = defaultdict(list)

        for row, (chunk_id, text) in enumerate(documents):
            tokens = tokenize(text)
            self.doc_ids.append(chunk_id)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((row, tf))

        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []

        n_docs = len(self.doc_ids)
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for row, tf in posting:
                dl = self.doc_len[row]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[row] += idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.doc_ids[row], score) for row, score in ranked]
