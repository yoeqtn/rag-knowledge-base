from __future__ import annotations

import pytest

from app.core.fusion import reciprocal_rank_fusion


def test_rrf_prefers_items_hit_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    order = [doc for doc, _ in fused]
    assert order[0] == "a"
    assert order[1] == "c"
    assert order[2] == "b"


def test_rrf_scores_are_positive_and_sorted():
    fused = reciprocal_rank_fusion([["x", "y"]], k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_rrf_weight_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
