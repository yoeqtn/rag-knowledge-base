from __future__ import annotations

from app.core.citations import extract_citations
from app.core.types import Chunk, RetrievedChunk


def _source(index: int, text: str, filename: str = "制度.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=f"c{index}",
            document_id="doc1",
            tenant="default",
            text=text,
            index=index,
            meta={"filename": filename, "role": "child"},
        ),
        score=0.9 - index * 0.1,
    )


def test_extracts_in_order_and_dedupes():
    sources = [_source(1, "报销流程说明"), _source(2, "住宿标准说明")]
    answer = "按流程提交申请 [2]，财务审核 [1]，重复引用 [2]。"
    citations = extract_citations(answer, sources)

    assert [c.index for c in citations] == [2, 1]
    assert citations[0].filename == "制度.md"
    assert citations[0].snippet.startswith("住宿标准")


def test_ignores_out_of_range_and_duplicate_numbers():
    sources = [_source(1, "内容")]
    answer = "引用 [9] 和 [0] 都不合法，只有 [1] 有效。"
    citations = extract_citations(answer, sources)
    assert [c.index for c in citations] == [1]


def test_no_citation_returns_empty():
    assert extract_citations("没有引用。", [_source(1, "内容")]) == []


def test_citation_dict_shape():
    sources = [_source(1, "很长的一段原文" * 30)]
    citation = extract_citations("见 [1]", sources)[0].to_dict()
    assert set(citation) == {
        "index",
        "document_id",
        "filename",
        "chunk_index",
        "snippet",
        "score",
    }
    assert len(citation["snippet"]) <= 120
