"""Class-aware optimizers. Only public observations enter these functions.

LM optimizes every allowed waveform from Sobol restarts. Phase projection uses
all inverse branches, followed by the same LM refinement. Both methods fit the
specified finite waveform family.
"""

import time
from fractions import Fraction

import numpy as np
import torch

from ..family import load_family
from ..io import sha256 as sha
from ..observations import load_public


def response(t, k, Q, derivative=False):
    value = torch.zeros_like(t)
    slope = torch.zeros_like(t)
    for j, c in enumerate(k):
        if c:
            w = 2 * torch.pi * (j + 1)
            value = value + (c / Q) * torch.cos(w * t)
            if derivative:
                slope = slope - (c / Q) * w * torch.sin(w * t)
    return (value, slope) if derivative else value


def lm(X, y, theta, k, Q, steps, norm_bound):
    """Batched damped Gauss-Newton with steps accepted by response MSE."""
    d = X.shape[1]
    damping = torch.full((len(theta),), 1e-3, dtype=X.dtype, device=X.device)
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    for _ in range(steps):
        pred, slope = response(theta @ X.T, k, Q, True)
        residual = pred - y
        loss = residual.square().mean(1)
        J = slope[:, :, None] * X[None]
        gram = J.transpose(1, 2) @ J / len(X)
        rhs = (J * residual[:, :, None]).mean(1)
        scale = gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-8)
        delta = torch.linalg.solve(
            gram + (damping * scale)[:, None, None] * eye, rhs[:, :, None]
        ).squeeze(-1)
        trial = theta - delta
        norm = trial.norm(dim=1, keepdim=True)
        trial = trial * torch.clamp(norm_bound / norm.clamp_min(1e-10), max=1)
        newloss = (response(trial @ X.T, k, Q) - y).square().mean(1)
        accept = newloss < loss
        theta = torch.where(accept[:, None], trial, theta)
        damping = torch.where(accept, damping * 0.4, damping * 5).clamp(1e-10, 1e10)
    return theta


def inverse_phases(y, k, Q):
    # Cubic/small-degree companion eigenproblems, with all real unit roots retained.
    coeff = np.array([0] + list(k), dtype=float) / Q
    phases = np.zeros((len(y), 2 * len(k)))
    mask = np.zeros_like(phases, dtype=bool)
    deriv = np.polynomial.chebyshev.chebder(coeff)
    crit = np.polynomial.chebyshev.chebroots(deriv)
    extrema = np.r_[-1.0, 1.0, [c.real for c in crit if abs(c.imag) < 1e-8 and -1 < c.real < 1]]
    extrema_y = np.polynomial.chebyshev.chebval(extrema, coeff)
    for i, v in enumerate(y):
        c = coeff.copy()
        c[0] = -v
        roots = np.polynomial.chebyshev.chebroots(c)
        real = sorted(
            r.real for r in roots if abs(r.imag) < 1e-7 and -1 - 1e-10 <= r.real <= 1 + 1e-10
        )
        if not real:
            # The observed label may lie outside this candidate waveform's range.
            real = [extrema[np.argmin(abs(extrema_y - v))]]
        vals = np.arccos(np.clip(real, -1, 1)) / (2 * np.pi)
        vals = np.r_[vals, -vals]
        phases[i, : len(vals)] = vals
        mask[i, : len(vals)] = True
    return phases, mask


def phase_projection(X, y, theta, k, Q, steps=32, temperature=0.08):
    phases, mask = inverse_phases(y.cpu().numpy(), k, Q)
    a = torch.tensor(phases, device=X.device, dtype=X.dtype)[None]
    valid = torch.tensor(mask, device=X.device)[None]
    projector = torch.linalg.solve(
        X.T @ X + 1e-9 * torch.eye(X.shape[1], device=X.device, dtype=X.dtype), X.T
    )
    for temp in np.geomspace(temperature, 0.002, steps):
        z = theta @ X.T
        targets = a + torch.round(z[:, :, None] - a)
        logits = -0.5 * ((targets - z[:, :, None]) / temp).square()
        probs = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        target = (probs * targets).sum(-1)
        theta = target @ projector.T
    return theta


def fit(
    public_path,
    family_path,
    method="multistart-lm",
    starts=128,
    steps=60,
    seed=8123,
    device="cpu",
    oracle_k=None,
):
    t0 = time.perf_counter()
    pub, data = load_public(public_path)
    _, family = load_family(family_path)
    keys = sorted(family) if oracle_k is None else [tuple(oracle_k)]
    _, _, xn, yn = data["train"]
    _, _, xv, yv = data["validation"]
    torch.set_num_threads(2)
    X = torch.tensor(xn, dtype=torch.float64, device=device)
    y = torch.tensor(yn, dtype=torch.float64, device=device)
    V = torch.tensor(xv, dtype=torch.float64, device=device)
    v = torch.tensor(yv, dtype=torch.float64, device=device)
    d = X.shape[1]
    u = (
        torch.quasirandom.SobolEngine(d + 1, scramble=True, seed=seed)
        .draw(starts)
        .to(device=device, dtype=X.dtype)
    )
    z = torch.erfinv((2 * u[:, :d] - 1).clamp(-0.999999, 0.999999))
    z = z / z.norm(dim=1, keepdim=True)
    z = z * torch.where(z[:, :1] < 0, -1.0, 1.0)
    init = z * (1 + (pub["norm_bound"] - 1) * u[:, d:])
    # Supplement direction restarts by a link-free spectral estimate at five norms.
    eig, vec = np.linalg.eigh((xn * (yn - yn.mean())[:, None]).T @ xn / len(xn))
    direction = vec[:, np.argmax(abs(eig))]
    spectral = torch.tensor(
        direction[None] * np.linspace(1, pub["norm_bound"], 5)[:, None],
        device=device,
        dtype=X.dtype,
    )
    init = torch.cat((init, spectral))
    attempts = []
    best = None
    # Explore a fixed subset, then refine candidates on all training rows.
    nsub = min(128, len(X))
    Xsub, ysub = X[:nsub], y[:nsub]
    with torch.no_grad():
        for k in keys:
            theta = init.clone()
            if method == "phase-projection":
                theta = torch.cat(
                    [
                        phase_projection(Xsub, ysub, theta, k, pub["Q"], temperature=t)
                        for t in (0.08, 0.25)
                    ]
                )
            theta = lm(Xsub, ysub, theta, k, pub["Q"], steps, pub["norm_bound"])
            losses = (response(theta @ X.T, k, pub["Q"]) - y).square().mean(1)
            ids = losses.argsort()[: min(12, len(theta))]
            theta = lm(X, y, theta[ids], k, pub["Q"], 30, pub["norm_bound"])
            trainloss = (response(theta @ X.T, k, pub["Q"]) - y).square().mean(1)
            valloss = (response(theta @ V.T, k, pub["Q"]) - v).square().mean(1)
            idx = int(valloss.argmin())
            vl = float(valloss[idx])
            model = dict(
                k=list(k),
                Q=pub["Q"],
                theta=[str(Fraction(float(t))) for t in theta[idx].cpu().numpy()],
            )
            attempts.append(dict(k=list(k), validation_mse=vl, train_mse=float(trainloss[idx])))
            if best is None or vl < best[0]:
                best = (vl, model)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return dict(
        method=method + ("-oracle-link" if oracle_k is not None else ""),
        status="FITTED",
        model=best[1],
        best_validation_mse=best[0],
        seconds=time.perf_counter() - t0,
        candidates=len(keys),
        starts=starts,
        spectral_starts=5,
        steps=steps,
        subset_rows=nsub,
        finalists_per_waveform=12,
        final_refinement_steps=30,
        device=device,
        dtype="float64",
        attempts=attempts,
        public_sha256=sha(public_path),
        family_sha256=sha(family_path),
        uses_oracle=oracle_k is not None,
    )
