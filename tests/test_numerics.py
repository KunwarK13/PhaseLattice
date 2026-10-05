from fractions import Fraction

import numpy as np
import pytest

from phaselattice.screening import response_range, screen_candidates


def test_exact_range_and_rounding_boundary():
    lower, upper, _ = response_range((1, 3), 4)
    assert lower <= Fraction(-73, 96)
    assert upper >= 1
    assert abs(float(lower) + 73 / 96) < 1e-14
    public = dict(Q=4, B=40)
    family = {(1, 3): None, (4,): None}
    data = {"train": (None, [lower - Fraction(1, 2**41), upper], None, None)}
    retained, _ = screen_candidates(public, data, family)
    assert len(retained) == 2
    data["train"] = (None, [Fraction(-9, 10), Fraction(1)], None, None)
    retained, _ = screen_candidates(public, data, family)
    assert retained == [(4,)]


def test_profiled_jacobian():
    torch = pytest.importorskip("torch")
    from phaselattice.baselines.variable_projection import profile

    rng = np.random.default_rng(819)
    inputs, outputs, weights = [
        torch.tensor(a, dtype=torch.float64)
        for a in (rng.normal(size=(53, 4)), rng.normal(size=53), rng.normal(size=(2, 4)))
    ]
    _, _, _, jacobian, _ = profile(inputs, outputs, weights, 3, True)
    h = 1e-6
    for j in range(4):
        delta = torch.zeros_like(weights)
        delta[:, j] = h
        a, _, _ = profile(inputs, outputs, weights + delta, 3)
        b, _, _ = profile(inputs, outputs, weights - delta, 3)
        np.testing.assert_allclose(jacobian[:, :, j], (a - b) / (2 * h), atol=2e-7, rtol=1e-6)


def test_continuous_waveform_local_recovery():
    torch = pytest.importorskip("torch")
    from phaselattice.baselines.variable_projection import optimize, profile

    inputs = torch.tensor(np.random.default_rng(915).normal(size=(256, 4)), dtype=torch.float64)
    weights = torch.tensor([[1.2, -0.6, 0.2, 0.3]], dtype=inputs.dtype)
    coefficients = torch.tensor([0.3, -0.15, 0.55], dtype=inputs.dtype)
    frequencies = 2 * torch.pi * torch.arange(1, 4, dtype=inputs.dtype)
    outputs = (torch.cos((weights @ inputs.T).T * frequencies) * coefficients).sum(-1)
    fitted = optimize(inputs, outputs, weights + 0.01, 3, 40, 2.0)
    prediction, beta, _ = profile(inputs, outputs, fitted, 3)
    np.testing.assert_allclose(fitted, weights, atol=1e-9, rtol=0)
    np.testing.assert_allclose(beta[0], coefficients, atol=1e-9, rtol=0)
    assert float((prediction[0] - outputs).square().mean()) < 1e-18


@pytest.mark.parametrize("method", ["multistart-lm", "phase-projection"])
def test_optimization_positive_control(tmp_path, method):
    pytest.importorskip("torch")
    from phaselattice import default_family
    from phaselattice.baselines.least_squares import fit
    from phaselattice.evaluation import score
    from phaselattice.synthetic import generate

    generate(tmp_path, default_family(), 1, 410099, k_override=(4,), n_train=128, n_val=64)
    result = fit(
        tmp_path / "observations.json", default_family(), method, starts=16, steps=35, oracle_k=(4,)
    )
    assert score(tmp_path, result)["recovered"]
