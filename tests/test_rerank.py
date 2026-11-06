from __future__ import annotations

import pytest

from app.core.rerank import (
    ApiReranker,
    NoopReranker,
    build_rerank_request,
    build_reranker,
    parse_rerank_response,
)


def test_siliconflow_request_is_flat():
    payload = build_rerank_request(
        "siliconflow", "BAAI/bge-reranker-v2-m3", "报销标准", ["a", "b"], 2
    )
    assert payload["query"] == "报销标准"
    assert payload["documents"] == ["a", "b"]
    assert payload["top_n"] == 2
    assert "input" not in payload


def test_dashscope_request_is_nested():
    payload = build_rerank_request("dashscope", "gte-rerank-v2", "报销标准", ["a", "b"], 5)
    assert payload["input"] == {"query": "报销标准", "documents": ["a", "b"]}
    assert payload["parameters"]["top_n"] == 2  # 不会超过文档数
    assert "query" not in payload


def test_parse_siliconflow_response_sorts_by_score():
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.31},
            {"index": 0, "relevance_score": 0.92},
        ]
    }
    assert parse_rerank_response(payload) == [(0, 0.92), (2, 0.31)]


def test_parse_dashscope_response():
    payload = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.4},
            ]
        }
    }
    assert parse_rerank_response(payload) == [(1, 0.8), (0, 0.4)]


def test_parse_response_with_data_key():
    payload = {"data": [{"index": 0, "score": 0.5}]}
    assert parse_rerank_response(payload) == [(0, 0.5)]


def test_parse_empty_response():
    assert parse_rerank_response({}) == []


def test_endpoint_for_siliconflow():
    reranker = ApiReranker("m", "https://api.siliconflow.cn/v1", "k", "siliconflow")
    assert reranker._endpoint() == "https://api.siliconflow.cn/v1/rerank"


def test_endpoint_for_dashscope():
    reranker = ApiReranker("m", "https://dashscope.aliyuncs.com/api/v1", "k", "dashscope")
    assert reranker._endpoint().endswith("/services/rerank/text-rerank/text-rerank")


def test_endpoint_keeps_full_dashscope_url():
    full = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    reranker = ApiReranker("m", full, "k", "dashscope")
    assert reranker._endpoint() == full


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        ApiReranker("m", "https://example.com", "k", "nope")


def test_missing_key_raises():
    with pytest.raises(ValueError):
        ApiReranker("m", "https://example.com", "", "siliconflow")


async def test_noop_reranker_keeps_order():
    reranker = NoopReranker()
    pairs = await reranker.rerank("q", ["a", "b", "c"], top_k=2)
    assert [index for index, _ in pairs] == [0, 1]


def test_build_reranker_disabled_returns_noop():
    assert isinstance(build_reranker(False, "dashscope", "m", "u", "k"), NoopReranker)


def test_build_reranker_enabled_returns_api():
    assert isinstance(build_reranker(True, "dashscope", "m", "u", "k"), ApiReranker)
