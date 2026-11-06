"""大模型调用。统一走 OpenAI 兼容协议，换服务商只改 .env。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

from app.core.types import Usage


@dataclass(slots=True)
class LLMDelta:
    text: str = ""
    usage: Usage | None = None


class ChatModel(Protocol):
    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]: ...

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]: ...


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        from openai import AsyncOpenAI

        if not api_key:
            raise ValueError("LLM_API_KEY 为空，请在 .env 中配置")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        kwargs = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        try:
            stream = await self._client.chat.completions.create(
                **kwargs, stream_options={"include_usage": True}
            )
        except TypeError:
            # 某些兼容服务不认 stream_options，退化成普通流式
            stream = await self._client.chat.completions.create(**kwargs)

        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                yield LLMDelta(
                    usage=Usage(
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    )
                )
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield LLMDelta(text=content)

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return content, Usage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class DisabledChatModel:
    """没配 API Key 时的占位实现：服务能启动，调用时给出明确报错。"""

    def __init__(self, reason: str = "未配置 LLM_API_KEY") -> None:
        self.reason = reason

    async def stream(self, messages, temperature=None, max_tokens=None):
        raise RuntimeError(self.reason)
        yield  # pragma: no cover  让函数保持 async generator 语义

    async def complete(self, messages, temperature=None, max_tokens=None):
        raise RuntimeError(self.reason)


def build_chat_model(settings) -> ChatModel:
    if settings.llm_api_key:
        return OpenAIChatModel(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    return DisabledChatModel()


class FakeChatModel:
    """测试用：不联网，按固定答案逐字返回，用来验证整条链路。"""

    def __init__(self, answer: str | None = None) -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    def _resolve(self, messages: Sequence[dict[str, str]]) -> str:
        # 多轮查询改写：直接回传最后一句话，模拟"改写后仍是独立问题"
        if messages and "改写" in messages[0].get("content", ""):
            last = messages[-1]["content"]
            marker = last.rfind("最后一句话：")
            return last[marker + len("最后一句话：") :].strip() if marker >= 0 else last

        if self.answer is not None:
            return self.answer
        # 默认答案里带上第一条参考资料，方便断言引用链路
        last = messages[-1]["content"] if messages else ""
        marker = last.find("[1]")
        snippet = last[marker : marker + 40] if marker >= 0 else "根据现有资料无法回答。"
        return f"这是测试回答。[1] {snippet}"

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        self.calls.append(list(messages))
        text = self._resolve(messages)
        for i in range(0, len(text), 8):
            yield LLMDelta(text=text[i : i + 8])
        yield LLMDelta(usage=Usage(prompt_tokens=120, completion_tokens=30))

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, Usage]:
        self.calls.append(list(messages))
        return self._resolve(messages), Usage(prompt_tokens=120, completion_tokens=30)
