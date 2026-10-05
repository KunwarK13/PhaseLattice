"""Executable periodic models reconstructed from observations."""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from string import Template

import numpy as np


@dataclass(frozen=True)
class PeriodicModel:
    coefficients: tuple[int, ...]
    denominator: int
    weights: tuple[Fraction, ...]

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "weights", tuple(Fraction(value) for value in self.weights))
        if (
            type(self.denominator) is not int
            or self.denominator < 1
            or not self.coefficients
            or any(type(v) is not int for v in self.coefficients)
            or sum(abs(v) for v in self.coefficients) != self.denominator
            or not self.weights
        ):
            raise ValueError("Invalid normalized periodic model")
        if not np.isfinite(self.theta).all():
            raise ValueError("Weights must be finite")

    @cached_property
    def theta(self) -> np.ndarray:
        values = np.array(self.weights, dtype=np.float64)
        values.flags.writeable = False
        return values

    @classmethod
    def from_record(cls, record: dict) -> "PeriodicModel":
        return cls(tuple(record["k"]), record["Q"], tuple(Fraction(v) for v in record["theta"]))

    def to_record(self) -> dict:
        return dict(
            k=list(self.coefficients), Q=self.denominator, theta=[str(v) for v in self.weights]
        )

    def _inputs(self, inputs) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != len(self.weights):
            raise ValueError(f"Expected inputs with final dimension {len(self.weights)}")
        if not np.isfinite(values).all():
            raise ValueError("Inputs must be finite")
        return values

    def predict(self, inputs) -> np.ndarray:
        phase = self._inputs(inputs) @ self.theta
        result = np.zeros_like(phase)
        for j, coefficient in enumerate(self.coefficients, start=1):
            if coefficient:
                result += coefficient * np.cos(2 * np.pi * j * phase)
        return result / self.denominator

    def gradient(self, inputs) -> np.ndarray:
        phase = self._inputs(inputs) @ self.theta
        slope = sum(
            -k * (2 * np.pi * j) * np.sin(2 * np.pi * j * phase) / self.denominator
            for j, k in enumerate(self.coefficients, start=1)
        )
        return np.asarray(slope)[..., None] * self.theta

    def hessian_vector_product(self, inputs, vector) -> np.ndarray:
        phase = self._inputs(inputs) @ self.theta
        projection = self._inputs(vector) @ self.theta
        curvature = sum(
            -k * (2 * np.pi * j) ** 2 * np.cos(2 * np.pi * j * phase) / self.denominator
            for j, k in enumerate(self.coefficients, start=1)
        )
        return (curvature * projection)[..., None] * self.theta

    def as_torch(self):
        """Return a float64 module with fixed weights and input autograd support."""
        import torch

        model = self

        class ReconstructedResponse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("theta", torch.tensor(model.theta.copy(), dtype=torch.float64))
                self.register_buffer(
                    "beta",
                    torch.tensor(model.coefficients, dtype=torch.float64) / model.denominator,
                )
                self.register_buffer(
                    "omega",
                    2
                    * torch.pi
                    * torch.arange(1, len(model.coefficients) + 1, dtype=torch.float64),
                )

            def forward(self, inputs):
                return (torch.cos((inputs @ self.theta)[..., None] * self.omega) * self.beta).sum(
                    -1
                )

        return ReconstructedResponse()

    def as_qiskit(self):
        """Return a circuit, observable and feature parameters for the recovered response."""
        from .quantum import reconstructed_circuit

        return reconstructed_circuit(self.coefficients, self.denominator, self.theta)

    def export(self, path: str | Path) -> None:
        """Write a standalone predictor requiring no PhaseLattice installation."""
        template = files("phaselattice").joinpath("templates/model.py.tmpl").read_text()
        source = Template(template).substitute(
            coefficients=repr(list(self.coefficients)),
            denominator=repr(self.denominator),
            weights=repr([str(v) for v in self.weights]),
        )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
