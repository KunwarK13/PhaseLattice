"""Validate the public observation boundary and preserve every supplied bit."""

from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .io import read_json

SCHEMA = "phaselattice.observations/v1"


class ObservationSplit(NamedTuple):
    exact_inputs: list[list[Fraction]]
    exact_outputs: list[Fraction]
    inputs: np.ndarray
    outputs: np.ndarray


def load_public(path: str | Path) -> tuple[dict, dict[str, ObservationSplit]]:
    record = read_json(path)
    fields = {"schema", "B", "D", "Q", "norm_bound", "train", "validation"}
    if not isinstance(record, dict) or set(record) != fields or record["schema"] != SCHEMA:
        raise ValueError("Expected a public observation record with no teacher metadata")
    precision = record["B"]
    if type(precision) is not int or not 1 <= precision <= 44:
        raise ValueError("Precision must be an integer from 1 to 44 fractional bits")
    if any(type(record[key]) is not int or record[key] < 1 for key in ("D", "Q")):
        raise ValueError("Degree and coefficient denominator must be positive integers")
    bound = record["norm_bound"]
    if type(bound) not in (int, float) or not np.isfinite(bound) or bound < 1:
        raise ValueError("The norm upper bound must be finite and at least one")
    splits = {}
    for name in ("train", "validation"):
        raw = record[name]
        if not isinstance(raw, dict) or set(raw) != {"X", "y"}:
            raise ValueError(f"Invalid {name} split")
        if not isinstance(raw["X"], list) or not isinstance(raw["y"], list):
            raise ValueError("Inputs and outputs must be arrays")
        if not raw["X"] or len(raw["X"]) != len(raw["y"]):
            raise ValueError("Every split needs equally many inputs and outputs")
        if not isinstance(raw["X"][0], list) or not raw["X"][0]:
            raise ValueError("Inputs must be nonempty vectors")
        dimension = len(raw["X"][0])
        if any(not isinstance(row, list) or len(row) != dimension for row in raw["X"]):
            raise ValueError("All input vectors must have the same dimension")
        scalars = [value for row in raw["X"] for value in row] + raw["y"]
        if any(type(value) is not int for value in scalars):
            raise ValueError("Observations must contain integer dyadic numerators")
        scale = 1 << precision
        exact_inputs = [[Fraction(value, scale) for value in row] for row in raw["X"]]
        exact_outputs = [Fraction(value, scale) for value in raw["y"]]
        inputs = np.array(exact_inputs, dtype=np.float64)
        outputs = np.array(exact_outputs, dtype=np.float64)
        exact_flat = (value for row in exact_inputs for value in row)
        if any(Fraction(float(a)) != b for a, b in zip(inputs.flat, exact_flat)):
            raise ValueError("An input cannot be represented losslessly in float64")
        if any(Fraction(float(a)) != b for a, b in zip(outputs, exact_outputs)):
            raise ValueError("An output cannot be represented losslessly in float64")
        splits[name] = ObservationSplit(exact_inputs, exact_outputs, inputs, outputs)
    if splits["train"].inputs.shape[1] != splits["validation"].inputs.shape[1]:
        raise ValueError("Training and validation dimensions differ")
    return record, splits
