from __future__ import annotations

from app.core.bm25 import BM25Index, tokenize


def test_tokenize_drops_punctuation():
    tokens = tokenize("报销流程，包括审核！")
    assert "报销" in tokens
    assert "，" not in tokens


def test_bm25_ranks_exact_keyword_match_first():
    index = BM25Index().build(
        [
            ("a", "差旅报销流程包括提交申请和财务审核"),
            ("b", "年假天数和请假审批流程"),
            ("c", "服务器巡检规范和告警处理"),
        ]
    )
    hits = index.search("报销 审核", top_k=3)
    assert hits
    assert hits[0][0] == "a"


def test_bm25_returns_empty_for_unknown_terms():
    index = BM25Index().build([("a", "报销流程")])
    assert index.search("量子计算", top_k=3) == []


def test_bm25_empty_index():
    assert BM25Index().build([]).search("任意", top_k=3) == []
