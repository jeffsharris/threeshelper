from __future__ import annotations

import numpy as np

from threes_rl.r15a_context_offline_gate import ece, rankdata, sign_accuracy, spearman


def test_rankdata_averages_ties() -> None:
    assert np.array_equal(rankdata(np.asarray([3.0, 1.0, 1.0, 2.0])), [3.0, 0.5, 0.5, 2.0])


def test_spearman_and_sign_accuracy() -> None:
    actual = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    predicted = np.asarray([-4.0, -2.0, 10.0, 2.0, 4.0])

    assert spearman(actual, predicted) < 1.0
    assert sign_accuracy(actual, predicted) == {"accuracy": 1.0, "informative": 4, "correct": 4}


def test_ece_is_zero_for_matching_two_bin_outcomes() -> None:
    probability = np.asarray([0.0, 0.0, 1.0, 1.0])
    outcome = probability.copy()
    weights = np.full(4, 0.25)

    assert ece(probability, outcome, weights) == 0.0
