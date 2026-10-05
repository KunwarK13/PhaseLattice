import importlib.util

import numpy as np
import pytest

from phaselattice import PeriodicModel
from phaselattice.quantum import quantum_labels


def exported_model(example, tmp_path):
    _, result = example
    result.model.export(tmp_path / "model.py")
    spec = importlib.util.spec_from_file_location("exported_response", tmp_path / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_predictions(example, tmp_path):
    _, result = example
    module = exported_model(example, tmp_path)
    inputs = np.random.default_rng(72).normal(size=(64, 3))
    np.testing.assert_allclose(module.predict(inputs), result.model.predict(inputs), atol=2e-14)


def test_derivatives_against_autograd(example, tmp_path):
    torch = pytest.importorskip("torch")
    _, result = example
    module = exported_model(example, tmp_path)
    rng = np.random.default_rng(414)
    inputs = rng.normal(size=(19, 3))
    vectors = rng.normal(size=inputs.shape)
    x = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    y = result.model.as_torch()(x)
    (gradient,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (hvp,) = torch.autograd.grad((gradient * torch.tensor(vectors)).sum(), x)
    for model in (module, result.model):
        np.testing.assert_allclose(
            model.gradient(inputs), gradient.detach(), atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            model.hessian_vector_product(inputs, vectors), hvp.detach(), atol=2e-11, rtol=1e-12
        )
    np.testing.assert_allclose(module.as_torch()(x).detach(), y.detach(), atol=2e-14)


def test_qiskit_harmonic_identity():
    pytest.importorskip("qiskit")
    phases = np.random.default_rng(899).normal(size=13) * 6
    for coefficients in ((-2, 0, 2), (1, -2, 1), (4,)):
        actual, _ = quantum_labels(coefficients, 4, phases)
        expected = sum(c * np.cos((j + 1) * phases) / 4 for j, c in enumerate(coefficients))
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_qiskit_feature_order_and_export(tmp_path):
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator

    model = PeriodicModel((1, -2, 1), 4, tuple(str(v) for v in np.linspace(-0.4, 0.5, 12)))
    inputs = np.random.default_rng(717).normal(size=(4, 12))
    model.export(tmp_path / "model.py")
    spec = importlib.util.spec_from_file_location("twelve_features", tmp_path / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for response in (model, module):
        circuit, observable, features = response.as_qiskit()
        pubs = [(circuit.assign_parameters(dict(zip(features, row))), observable) for row in inputs]
        actual = [float(r.data.evs) for r in StatevectorEstimator().run(pubs).result()]
        np.testing.assert_allclose(actual, model.predict(inputs), atol=2e-14, rtol=0)


def test_shots_are_independent_and_reproducible():
    pytest.importorskip("qiskit")
    phases = np.full(70, np.pi / 2)
    first, _ = quantum_labels((4,), 4, phases, 256, 811)
    second, _ = quantum_labels((4,), 4, phases, 256, 811)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) > 5
    assert abs(np.mean(first)) < 0.04
