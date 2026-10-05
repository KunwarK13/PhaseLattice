"""Replay reconstruction traces without invoking lattice reduction."""

from fractions import Fraction as F

import numpy as np

from . import algebra as pl
from .io import sha256 as sha
from .observations import load_public


def _verify_trace(public_path, result):
    """Replay row assignments and least-squares solution, not a uniqueness proof."""
    if result["status"] != "RECOVERED" or result.get("trace") is None:
        return dict(valid=False, reason="No recovery trace")
    if result["public_sha256"] != sha(public_path):
        return dict(valid=False, reason="Observation digest mismatch")
    pub, data = load_public(public_path)
    xs, ys, _, _ = data["train"]
    tr, model = (result["trace"], result["model"])
    th = [F(v) for v in model["theta"]]
    if (
        len(th) != len(xs[0])
        or model["Q"] != pub["Q"]
        or (not 1 <= len(model["k"]) <= pub["D"])
        or any((type(v) is not int for v in model["k"]))
        or (sum((abs(v) for v in model["k"])) != pub["Q"])
    ):
        return dict(valid=False, reason="Model violates public specification")
    n = len(tr["indices"])
    expected_rows = (
        len(th) + 1 if result["method"] == "phaselattice-single-relation" else 2 * len(th)
    )
    if (
        n != expected_rows
        or tr["chi"] not in (1, 2)
        or len(set(tr["indices"])) != n
        or any((len(tr[key]) != n for key in ("phases", "branches", "signs", "lifts")))
    ):
        return dict(valid=False, reason="Malformed trace")
    if np.linalg.norm(np.array(th, dtype=float)) > pub["norm_bound"] + 1e-06:
        return dict(valid=False, reason="Model violates public norm bound")
    residuals, zs, rows = ([], [], [])
    try:
        for i, phases, branch, sign, lift in zip(
            tr["indices"], tr["phases"], tr["branches"], tr["signs"], tr["lifts"]
        ):
            if (
                not 0 <= i < len(xs)
                or len(phases) != tr["chi"]
                or sign not in (-1, 1)
                or (not 0 <= branch < len(phases))
                or (type(lift) is not int)
            ):
                return dict(valid=False, reason="Invalid assignment")
            row = xs[i]
            if not F(tr["band"][0]) <= ys[i] <= F(tr["band"][1]):
                return dict(valid=False, reason="Selected label outside band")
            with pl.mp.workprec(pub["B"] + 64):
                for phase in phases:
                    a = F(phase)
                    if not 0 <= a <= F(1, 2):
                        return dict(valid=False, reason="Invalid folded phase")
                    value = pl.link_value(
                        model["k"], model["Q"], pl.mp.mpf(a.numerator) / a.denominator
                    )
                    tolerance = (2 * np.pi * pub["D"] + 2) * 2.0 ** (-pub["B"])
                    if abs(float(value) - float(ys[i])) > tolerance:
                        return dict(valid=False, reason="Invalid inverse phase")
            z = F(sign) * F(phases[branch]) + lift
            zs.append(z)
            rows.append(row)
            residuals.append(abs(sum((a * b for a, b in zip(row, th))) - z))
        if pl.sp.Matrix(rows).rank() != len(th):
            return dict(valid=False, reason="Selected inputs are rank deficient")
        normal = [
            sum((row[j] * (sum((a * b for a, b in zip(row, th))) - z) for row, z in zip(rows, zs)))
            for j in range(len(th))
        ]
        if any(normal):
            return dict(valid=False, reason="Normal equations failed")
        max_res = float(max(residuals))
        if max_res > 2.0 ** (-pub["B"] / 2):
            return dict(valid=False, reason="Large phase residual")
    except (ValueError, IndexError, ZeroDivisionError):
        return dict(valid=False, reason="Malformed rational data")
    return dict(
        valid=True,
        rows=n,
        max_phase_residual=max_res,
        statement="Consistent inverse phases and exact least-squares fit; not uniqueness",
    )


def verify_trace(public_path, result):
    """Check inverse phases, full-rank rational least squares, and validation fit."""
    try:
        if not isinstance(result, dict):
            return dict(valid=False, reason="Expected a reconstruction record")
        check = _verify_trace(public_path, result)
        if not check["valid"]:
            return check
        public, data = load_public(public_path)
        model = result["model"]
        xs, ys, _, _ = data["validation"]
        loss = pl.quantized_validation_loss(
            model["k"], model["Q"], [F(v) for v in model["theta"]], xs, ys, public["B"] + 64
        )
        if loss > 1e-8:
            return dict(
                valid=False,
                reason="Model does not fit validation observations",
                validation_mse=loss,
            )
        check["validation_mse"] = loss
        return check
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError):
        return dict(valid=False, reason="Malformed reconstruction record")
