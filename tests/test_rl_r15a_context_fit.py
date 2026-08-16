from __future__ import annotations

import numpy as np

from threes_rl.context_residual import ContextResidualModel, OUTPUT_NAMES
from threes_rl.r15a_context_fit import active_outputs, sigmoid, softmax


def test_active_outputs_selects_each_stage_head() -> None:
    model = ContextResidualModel(mode="board_stage_only")
    model.w1.fill(0.0)
    model.b1.fill(0.0)
    model.w2.fill(0.0)
    for stage in range(4):
        start = stage * len(OUTPUT_NAMES)
        model.b2[start : start + len(OUTPUT_NAMES)] = stage + np.arange(len(OUTPUT_NAMES)) / 100.0
    x = np.zeros((4, 64), dtype=np.float64)

    outputs = active_outputs(model, x, np.arange(4))

    assert outputs.shape == (4, len(OUTPUT_NAMES))
    assert np.allclose(outputs[:, 0], np.arange(4))


def test_softmax_and_sigmoid_are_finite_and_normalized() -> None:
    values = np.asarray([[1_000.0, 999.0, -1_000.0], [-1_000.0, -999.0, 1_000.0]])
    probabilities = softmax(values)

    assert np.all(np.isfinite(probabilities))
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert np.all(np.isfinite(sigmoid(values)))
    assert np.all((sigmoid(values) >= 0.0) & (sigmoid(values) <= 1.0))
