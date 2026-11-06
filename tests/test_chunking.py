from __future__ import annotations

from app.core.chunking import build_chunks

TEXT = "\n\n".join(
    [
        "第一段：公司差旅报销需要在出差结束后十五个工作日内提交申请。",
        "第二段：住宿标准按照城市等级划分，一线城市每晚不超过六百元。",
        "第三段：交通标准为高铁二等座或飞机经济舱，市内交通实报实销。",
        "第四段：特殊情况下产生的退票费和改签费，凭说明材料可以全额报销。",
    ]
)


def test_empty_text_returns_empty_list():
    assert build_chunks("   \n\n  ") == []


def test_fixed_strategy_respects_size():
    drafts = build_chunks(TEXT, strategy="fixed", chunk_size=50, chunk_overlap=10)
    assert drafts
    assert all(len(draft.text) <= 50 for draft in drafts)
    assert [d.index for d in drafts] == list(range(len(drafts)))


def test_recursive_strategy_keeps_content():
    drafts = build_chunks(TEXT, strategy="recursive", chunk_size=80, chunk_overlap=10)
    joined = "".join(draft.text for draft in drafts)
    assert "十五个工作日" in joined
    assert "六百元" in joined
    assert all(len(draft.text) <= 80 for draft in drafts)


def test_parent_child_links_every_child_to_a_parent():
    drafts = build_chunks(
        TEXT,
        strategy="parent_child",
        chunk_size=400,
        chunk_overlap=20,
        parent_size=120,
        child_size=60,
    )
    parents = [d for d in drafts if d.meta["role"] == "parent"]
    children = [d for d in drafts if d.meta["role"] == "child"]

    assert parents and children
    parent_indexes = {p.index for p in parents}
    assert all(child.parent_local_id in parent_indexes for child in children)
    assert all(len(child.text) <= 60 for child in children)


def test_parent_child_guards_too_small_parent_size():
    drafts = build_chunks(
        TEXT, strategy="parent_child", parent_size=10, child_size=60, chunk_overlap=0
    )
    parents = [d for d in drafts if d.meta["role"] == "parent"]
    assert parents
    assert all(len(p.text) >= 1 for p in parents)


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        build_chunks(TEXT, strategy="magic")  # type: ignore[arg-type]
