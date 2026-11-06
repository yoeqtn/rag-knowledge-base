"""从模型输出里解析 [n] 引用，映射回具体文档块。

这是 RAG 的"可验证性"：用户能点开 [1] 看到原文，
而不是只能相信模型的嘴。面试时这是很加分的点。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.core.types import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SNIPPET_LEN = 120


@dataclass(slots=True)
class Citation:
    index: int
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }


def extract_citations(answer: str, sources: Sequence[RetrievedChunk]) -> list[Citation]:
    """按出现顺序去重；越界的编号直接忽略，不做静默映射。"""
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if index in seen or not (1 <= index <= len(sources)):
            continue
        seen.add(index)
        source = sources[index - 1]
        text = source.text_for_prompt.replace("\n", " ").strip()
        citations.append(
            Citation(
                index=index,
                document_id=source.chunk.document_id,
                filename=str(source.chunk.meta.get("filename", "未知文档")),
                chunk_index=source.chunk.index,
                snippet=text[:_SNIPPET_LEN],
                score=source.score,
            )
        )

    return citations


def referenced_text(answer: str, sources: Sequence[RetrievedChunk]) -> str:
    """调试用：把答案里引用到的原文拼起来。"""
    parts = []
    for citation in extract_citations(answer, sources):
        parts.append(sources[citation.index - 1].text_for_prompt)
    return "\n\n".join(parts)
