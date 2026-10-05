"""Inference from public observations and a finite waveform family."""

import time

import numpy as np

from . import algebra as pl
from .family import load_family
from .io import sha256 as sha
from .observations import load_public


def recover(public_path, family_path, relations="full", accept_mse=1e-08, range_screen=True):
    started = time.perf_counter()
    public, data = load_public(public_path)
    meta, family = load_family(family_path)
    if (public["D"], public["Q"]) != (meta["D"], meta["Q"]):
        raise ValueError("Family differs from public specification")
    xs, ys, X, y = data["train"]
    xv, yv, _, _ = data["validation"]
    d, B = (X.shape[1], public["B"])
    m = 2 * d if relations == "full" else d + 1
    if relations not in ("full", "single"):
        raise ValueError("Unknown relation mode")
    screening = None
    keys = set(family)
    if range_screen:
        from .screening import screen_candidates

        keys, screening = screen_candidates(public, data, family)
        keys = set(keys)
    attempts, hypotheses = ([], [])
    total_reduction = 0.0
    for k, bands in family.items():
        if k not in keys:
            continue
        for b in bands:
            t = time.perf_counter()
            xr, lists, idx = pl.select_and_invert(b, xs, ys, m, B)
            entry = dict(k=list(k), chi=b.chi, band=[str(v) for v in b.J_zero], selected=len(idx))
            if len(xr) < m:
                entry.update(status="INSUFFICIENT_SELECTED_ROWS", seconds=time.perf_counter() - t)
                attempts.append(entry)
                continue
            res = pl.decode_mr(xr, lists, b.chi, B, public["norm_bound"])
            total_reduction += res.reduction_seconds
            entry.update(
                status=res.status,
                seconds=time.perf_counter() - t,
                reduction_seconds=res.reduction_seconds,
            )
            if res.status == "OK":
                if np.linalg.norm(res.theta_hat) > public["norm_bound"] + 1e-06:
                    entry["status"] = "OUTSIDE_PUBLIC_NORM_BOUND"
                else:
                    loss = pl.quantized_validation_loss(k, b.Q, res.theta_exact, xv, yv, B + 64)
                    trace = dict(
                        indices=idx,
                        phases=[[str(v) for v in row] for row in lists],
                        branches=res.branches,
                        signs=res.signs,
                        lifts=res.lifts,
                        band=[str(v) for v in b.J_zero],
                        chi=b.chi,
                    )
                    model = dict(k=list(k), Q=b.Q, theta=[str(v) for v in res.theta_exact])
                    hypotheses.append(dict(model=model, validation_mse=loss, trace=trace))
                    entry["validation_mse"] = loss
            attempts.append(entry)
    best = min(hypotheses, key=lambda h: h["validation_mse"]) if hypotheses else None
    accepted = best is not None and best["validation_mse"] <= accept_mse
    return dict(
        schema="phaselattice.reconstruction/v1",
        method="phaselattice" if relations == "full" else "phaselattice-single-relation",
        status="RECOVERED" if accepted else "UNRESOLVED",
        model=best["model"] if accepted else None,
        trace=best["trace"] if accepted else None,
        best_validation_mse=best["validation_mse"] if best else None,
        acceptance_mse=accept_mse,
        candidates=len(family),
        bands=len(attempts),
        successful_lattice_hypotheses=len(hypotheses),
        attempts=attempts,
        seconds=time.perf_counter() - started,
        reduction_seconds=total_reduction,
        family_compile_seconds=meta["seconds"],
        selected_rows=m,
        screening=screening,
        raw_train_rows=len(xs),
        raw_validation_rows=len(xv),
        public_sha256=sha(public_path),
        family_sha256=sha(family_path),
    )
