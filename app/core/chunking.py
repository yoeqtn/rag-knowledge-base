"""切块策略。

知识库问答的效果，一半取决于切块。
这里实现三种策略，面试时能讲清楚"为什么选它"比代码本身更重要：

1. fixed        —— 定长 + 重叠。最简单，但会把句子/段落切断。
2. recursive    —— 按段落 -> 句子 -> 硬切 逐级下降，尽量在语义边界断开。
3. parent_child —— 小块用于检索（精确），父块用于生成（上下文完整）。
                   这是本项目默认策略，也是实际生产里最常用的一档。
"""

from __future__ import annotations

import re
from typing import Literal

from app.core.types import ChunkDraft

Strategy = Literal["fixed", "recursive", "parent_child"]

# 按中文/英文句子结束符切分，保留标点
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def build_chunks(
    text: str,
    strategy: Strategy = "parent_child",
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    parent_size: int = 1200,
    child_size: int = 300,
) -> list[ChunkDraft]:
    """把整篇文档切成 ChunkDraft 列表。index 在文档内唯一且递增。"""
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    if strategy == "fixed":
        return _fixed(text, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return _recursive(text, chunk_size, chunk_overlap)
    if strategy == "parent_child":
        return _parent_child(text, parent_size, child_size, chunk_overlap)
    raise ValueError(f"未知切块策略: {strategy}")


# --------------------------------------------------------------------------- #
# 三种策略
# --------------------------------------------------------------------------- #
def _fixed(text: str, size: int, overlap: int) -> list[ChunkDraft]:
    step = max(1, size - overlap)
    drafts: list[ChunkDraft] = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + size].strip()
        if piece:
            drafts.append(
                ChunkDraft(text=piece, index=i, meta={"role": "child", "strategy": "fixed"})
            )
    return drafts


def _recursive(text: str, size: int, overlap: int) -> list[ChunkDraft]:
    pieces = _merge_units(text, size, overlap)
    return [
        ChunkDraft(text=p, index=i, meta={"role": "child", "strategy": "recursive"})
        for i, p in enumerate(pieces)
    ]


def _parent_child(
    text: str, parent_size: int, child_size: int, overlap: int
) -> list[ChunkDraft]:
    # 父块必须明显大于子块，否则这个策略没有意义
    parent_size = max(parent_size, child_size * 2)
    parent_overlap = min(overlap, child_size // 3)

    drafts: list[ChunkDraft] = []
    index = 0
    for parent_text in _merge_units(text, parent_size, overlap=0):
        parent_index = index
        index += 1
        drafts.append(
            ChunkDraft(
                text=parent_text,
                index=parent_index,
                meta={"role": "parent", "strategy": "parent_child"},
            )
        )
        for child_text in _merge_units(parent_text, child_size, parent_overlap):
            drafts.append(
                ChunkDraft(
                    text=child_text,
                    index=index,
                    parent_local_id=parent_index,
                    meta={"role": "child", "strategy": "parent_child"},
                )
            )
            index += 1
    return drafts


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _units(text: str, max_unit: int) -> list[str]:
    """把文本拆成"不可再拆的最小语义单元"：段落 -> 句子 -> 硬切。

    max_unit 必须跟随目标 chunk_size，否则小块策略会被超长段落撑破。
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_unit:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_RE.findall(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_unit:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_unit))
    return units


def _merge_units(text: str, size: int, overlap: int) -> list[str]:
    """贪心合并相邻单元，直到接近 size；新块开头带上上一块的尾巴做重叠。"""
    units = _units(text, max_unit=max(1, size))
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n{unit}" if tail else unit
            if len(current) > size:
                # 加了重叠就超了，宁可丢掉重叠
                current = unit
        else:
            current = unit

    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    step = max(1, size)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]
