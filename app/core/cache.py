"""语义缓存。

普通 KV 缓存只能命中"一模一样"的问题；语义缓存用向量相似度，
"报销流程是什么" 和 "怎么报销" 可以命中同一条缓存。

代价：阈值设低了会返回错误答案，所以默认 0.95，并且只对无多轮上下文的
单轮问题启用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CachedAnswer:
    question: str
    answer: str
    citations: list[dict]
    created_at: float


class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.95,
        max_entries: int = 512,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._vectors: list[np.ndarray] = []
        self._entries: list[CachedAnswer] = []
        self.hits = 0
        self.misses = 0

    def _evict(self) -> None:
        now = time.time()
        keep = [
            (vec, entry)
            for vec, entry in zip(self._vectors, self._entries)
            if now - entry.created_at < self.ttl
        ]
        self._vectors = [item[0] for item in keep][-self.max_entries :]
        self._entries = [item[1] for item in keep][-self.max_entries :]

    def get(self, vector: list[float]) -> CachedAnswer | None:
        if not self._entries:
            return None
        query = np.asarray(vector, dtype=np.float32)
        matrix = np.vstack(self._vectors)
        scores = matrix @ query
        best = int(np.argmax(scores))
        if float(scores[best]) >= self.threshold:
            self.hits += 1
            return self._entries[best]
        self.misses += 1
        return None

    def put(self, vector: list[float], question: str, answer: str, citations: list[dict]) -> None:
        self._vectors.append(np.asarray(vector, dtype=np.float32))
        self._entries.append(
            CachedAnswer(
                question=question,
                answer=answer,
                citations=citations,
                created_at=time.time(),
            )
        )
        self._evict()

    def clear(self) -> None:
        self._vectors.clear()
        self._entries.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
