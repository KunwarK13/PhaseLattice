"""Synthetic teachers. Evaluator files are never passed to reconstruction."""

import time
from pathlib import Path

import numpy as np

from . import sampling as pl
from .family import load_family
from .io import sha256 as sha
from .io import write_json
from .quantum import quantum_labels


def generate(
    directory,
    family_path,
    d,
    seed,
    gamma=1.5,
    B=40,
    n_train=1024,
    n_val=128,
    n_test=2048,
    provider="analytic",
    shots=0,
    k_override=None,
    norm_bound=2.0,
):
    start = time.perf_counter()
    if d < 1 or B < 1 or B > 44 or min(n_train, n_val, n_test) < 1 or not 1 <= gamma <= norm_bound:
        raise ValueError("Invalid teacher dimensions, precision, observation counts or norm")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta, family = load_family(family_path)
    rng = np.random.default_rng(seed)
    keys = sorted(family)
    k = tuple(k_override) if k_override is not None else keys[rng.integers(len(keys))]
    if k not in family:
        raise ValueError("Teacher outside family")
    inst = pl.make_direction(d, gamma, rng, B + 64)
    xs, ys, ws = pl.draw_observations(inst, k, meta["Q"], n_train + n_val, B, rng)
    provider_meta = {}
    if provider == "qiskit":
        with pl.mp.workprec(B + 64):
            phases = np.array([float(2 * pl.mp.pi * w) for w in ws])
            expected = np.array([float(pl.link_value(k, meta["Q"], w)) for w in ws])
        labels, provider_meta = quantum_labels(k, meta["Q"], phases, shots, seed + 317)
        provider_meta["max_error_vs_analytic"] = float(np.max(np.abs(labels - expected)))
        with pl.mp.workprec(B + 64):
            ys = [pl.round_dyadic(pl.mp.mpf(float(v)), B) for v in labels]
    elif provider != "analytic":
        raise ValueError("Unknown provider")
    scale = 1 << B
    rows = dict(X=[[int(v * scale) for v in row] for row in xs], y=[int(v * scale) for v in ys])
    if not 1 <= gamma <= norm_bound:
        raise ValueError("Teacher violates public norm bounds")
    public = dict(
        schema="phaselattice.observations/v1",
        B=B,
        D=meta["D"],
        Q=meta["Q"],
        norm_bound=norm_bound,
        train={key: val[:n_train] for key, val in rows.items()},
        validation={key: val[n_train:] for key, val in rows.items()},
    )
    write_json(directory / "observations.json", public)
    Xt = rng.standard_normal((n_test, d))
    yt = pl.predict(k, meta["Q"], inst.theta_f, Xt)
    np.savez_compressed(directory / "test.npz", X=Xt, y=yt)
    truth = dict(
        k=list(k),
        Q=meta["Q"],
        theta=inst.theta_f.tolist(),
        d=d,
        gamma=gamma,
        seed=seed,
        provider=provider,
        provider_metadata=provider_meta,
        public_sha256=sha(directory / "observations.json"),
        seconds=time.perf_counter() - start,
    )
    write_json(directory / "reference.json", truth)
    return truth
