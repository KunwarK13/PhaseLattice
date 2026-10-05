"""Evaluation against hidden parameters and independent test observations."""

from pathlib import Path

import numpy as np

from .io import read_json
from .model import PeriodicModel


def score(directory: str | Path, result: dict) -> dict:
    directory = Path(directory)
    reference = read_json(directory / "reference.json")
    with np.load(directory / "test.npz") as test:
        inputs, outputs = test["X"], test["y"]
    if result["model"] is None:
        return dict(
            recovered=False,
            prediction_success=False,
            exact_link=False,
            direction_error=None,
            test_mse=None,
            loss_over_variance=None,
        )
    model = PeriodicModel.from_record(result["model"])
    truth = np.array(reference["theta"])
    error = min(np.linalg.norm(model.theta - truth), np.linalg.norm(model.theta + truth))
    mse = float(np.mean((model.predict(inputs) - outputs) ** 2))
    relative_mse = mse / float(np.var(outputs))
    same_link = list(model.coefficients) == reference["k"] and model.denominator == reference["Q"]
    return dict(
        recovered=bool(same_link and error < 1e-4 and relative_mse < 1e-6),
        prediction_success=bool(relative_mse < 1e-6),
        exact_link=same_link,
        direction_error=float(error),
        test_mse=mse,
        loss_over_variance=relative_mse,
    )


def relative_error(prediction, target) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1e-30))


def reference_derivatives(reference: dict, inputs: np.ndarray, vectors: np.ndarray):
    """Differentiate the hidden equation independently with PyTorch autograd."""
    import torch

    x = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(reference["theta"], dtype=torch.float64)
    beta = torch.tensor(reference["k"], dtype=torch.float64) / reference["Q"]
    omega = 2 * torch.pi * torch.arange(1, len(beta) + 1, dtype=torch.float64)
    y = (torch.cos((x @ theta)[:, None] * omega) * beta).sum(-1)
    (gradient,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (hvp,) = torch.autograd.grad((gradient * torch.tensor(vectors)).sum(), x)
    return y.detach().numpy(), gradient.detach().numpy(), hvp.detach().numpy()


def quantum_roundtrip(model: PeriodicModel, reference: dict, inputs: np.ndarray) -> dict:
    from qiskit.primitives import StatevectorEstimator

    from .quantum import quantum_labels

    circuit, observable, features = model.as_qiskit()
    pubs = [(circuit.assign_parameters(dict(zip(features, row))), observable) for row in inputs]
    replica = np.array(
        [float(result.data.evs) for result in StatevectorEstimator().run(pubs).result()]
    )
    theta = np.array(reference["theta"])
    original, _ = quantum_labels(reference["k"], reference["Q"], 2 * np.pi * (inputs @ theta))
    step = 1e-4
    gradient = np.zeros_like(inputs)
    for j in range(inputs.shape[1]):
        points = []
        for scale in (-2, -1, 1, 2):
            perturbed = inputs.copy()
            perturbed[:, j] += scale * step
            points.append(perturbed)
        values, _ = quantum_labels(
            reference["k"], reference["Q"], 2 * np.pi * (np.concatenate(points) @ theta)
        )
        a, b, c, d = values.reshape(4, len(inputs))
        gradient[:, j] = (a - 8 * b + 8 * c - d) / (12 * step)
    return dict(
        points=len(inputs),
        max_circuit_difference=float(np.max(abs(replica - original))),
        gradient_relative_rms_error=relative_error(model.gradient(inputs), gradient),
        max_gradient_difference=float(np.max(abs(model.gradient(inputs) - gradient))),
        difference_step=step,
        original_circuit_evaluations=len(inputs) * (1 + 4 * inputs.shape[1]),
    )
