"""Compile and load certified finite families of periodic waveforms."""

import time
from dataclasses import asdict
from fractions import Fraction
from importlib.resources import files
from pathlib import Path

import numpy as np

from . import algebra
from .io import content_digest, read_json, sha256, write_json

SCHEMA = "phaselattice.family/v1"


def default_family() -> Path:
    return Path(str(files("phaselattice").joinpath("resources/family.json")))


def compile_family(path: str | Path, degree: int = 3, denominator: int = 4) -> dict:
    if type(degree) is not int or type(denominator) is not int or min(degree, denominator) < 1:
        raise ValueError("Degree and denominator must be positive integers")
    started = time.perf_counter()
    family, census = algebra.certified_family(degree, denominator, chi_max=2)
    bands = []
    for key in sorted(family):
        for band in family[key]:
            record = asdict(band)
            for name in ("comp", "J_plus", "J_zero", "J_minus"):
                record[name] = [str(value) for value in record[name]]
            record["Delta"] = str(band.Delta)
            for name in ("rho", "alpha", "L", "min_h2"):
                if not np.isfinite(record[name]):
                    record[name] = None
            bands.append(record)
    result = dict(
        schema=SCHEMA,
        D=degree,
        Q=denominator,
        census=census,
        bands=bands,
        bands_sha256=content_digest(bands),
        compiler_sha256=sha256(algebra.__file__),
        seconds=time.perf_counter() - started,
    )
    write_json(path, result)
    return result


def load_family(path: str | Path | None = None) -> tuple[dict, dict]:
    record = read_json(default_family() if path is None else path)
    if record.get("schema") != SCHEMA or record.get("bands_sha256") != content_digest(
        record["bands"]
    ):
        raise ValueError("Invalid family schema or band digest")
    family = {}
    for raw in record["bands"]:
        row = dict(raw)
        row["k"] = tuple(row["k"])
        for name in ("comp", "J_plus", "J_zero", "J_minus"):
            row[name] = tuple(Fraction(value) for value in row[name])
        row["Delta"] = Fraction(row["Delta"])
        for name in ("rho", "alpha", "L", "min_h2"):
            if row[name] is None:
                row[name] = float("nan")
        band = algebra.Band(**row)
        if (
            band.chi not in (1, 2)
            or band.status != "exact:chi<=2"
            or band.Q != record["Q"]
            or len(band.k) > record["D"]
            or any(type(v) is not int for v in band.k)
            or sum(abs(v) for v in band.k) != band.Q
        ):
            raise ValueError("Band violates the family specification")
        family.setdefault(band.k, []).append(band)
    return record, family
