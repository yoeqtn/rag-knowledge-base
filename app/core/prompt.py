"""Prompt 模板。

原则：
1. 只允许基于检索到的资料回答，越权回答是 RAG 最大的坑
2. 强制编号引用，方便前端做溯源、也方便评测
3. 资料不足时明确说"不知道"，比编一个答案安全
"""

from __future__ import annotations

from typing import Sequence

from app.core.types import RetrievedChunk

SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。

回答规则：
1. 只能依据【参考资料】中的内容回答，禁止使用资料以外的知识。
2. 如果资料不足以回答问题，直接回答"根据现有资料无法回答"，并说明还缺少什么信息。
3. 每一条结论后面必须标注来源编号，格式为 [1] 或 [1][2]，编号必须来自参考资料。
4. 不要输出参考资料中不存在的事实、数字、人名或链接。
5. 用中文回答；问题复杂时分点作答，简洁优先。"""

REWRITE_PROMPT = """把用户在多轮对话中的最后一句话改写成一个不依赖上下文、可以独立检索的问题。
只输出改写后的问题本身，不要解释，不要加引号。
如果最后一句话本身已经是完整问题，原样输出。"""

NO_CONTEXT_ANSWER = "根据现有资料无法回答这个问题，知识库中没有检索到相关内容。"


def build_context(sources: Sequence[RetrievedChunk]) -> str:
    if not sources:
        return "（没有检索到任何资料）"

    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        filename = source.chunk.meta.get("filename", "未知文档")
        blocks.append(
            f"[{index}] 来源：{filename}（第 {source.chunk.index} 块）\n{source.text_for_prompt}"
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    sources: Sequence[RetrievedChunk],
    history: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 只带最近 3 轮，避免 context 无限膨胀
    for turn in list(history or [])[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"【参考资料】\n{build_context(sources)}\n\n"
        f"【问题】\n{question}\n\n"
        f"请依据参考资料回答，并在句末标注来源编号。"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def build_rewrite_messages(
    question: str, history: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    dialogue = "\n".join(f"{turn['role']}: {turn['content']}" for turn in list(history)[-6:])
    return [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"对话历史：\n{dialogue}\n\n最后一句话：{question}"},
    ]
