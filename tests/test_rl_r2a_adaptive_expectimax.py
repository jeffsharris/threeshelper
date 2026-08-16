from __future__ import annotations

from threes_rl.r2a_adaptive_expectimax import milestone_for_built_max, normalized_margin


def test_milestone_for_built_max_targets_next_level() -> None:
    assert milestone_for_built_max(384) is None
    assert milestone_for_built_max(768) == 1536
    assert milestone_for_built_max(1536) == 3072
    assert milestone_for_built_max(3072) is None


def test_normalized_margin_is_scale_stable() -> None:
    assert normalized_margin([(0, 100.0), (1, 98.0)]) == 0.02
    assert normalized_margin([(0, 200.0), (1, 196.0)]) == 0.02
    assert normalized_margin([(0, 5.0)]) == 1.0
