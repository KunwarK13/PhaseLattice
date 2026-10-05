"""Variable projection for unknown harmonic coefficients and input weights.

At each proposed direction, solve for D real Fourier coefficients. The exact
Jacobian of the regularized least-squares solution is used in batched LM.
Selected directions are transferred across harmonics and refined against the
known discrete family.
Only public observations and the family enter this module.
"""

import time
from fractions import Fraction

import numpy as np
import torch

from ..family import load_family
from ..io import sha256 as sha
from ..observations import load_public
from .least_squares import lm, response


def profile(X, y, theta, degree, jacobian=False, ridge=1e-12):
    """Profiled predictions and their exact direction Jacobian (including ridge)."""
    omega = 2 * torch.pi * torch.arange(1, degree + 1, dtype=X.dtype, device=X.device)
    phase = (theta @ X.T)[..., None] * omega
    C = phase.cos()
    Ct = C.transpose(1, 2)
    G = Ct @ C / len(X) + ridge * torch.eye(degree, dtype=X.dtype, device=X.device)
    beta = torch.linalg.solve(G, (Ct @ y)[:, :, None] / len(X)).squeeze(-1)
    prediction = (C * beta[:, None]).sum(-1)
    residual = prediction - y
    objective = residual.square().mean(1) + ridge * beta.square().sum(1)
    if not jacobian:
        return prediction, beta, objective
    dC_dt = -omega * phase.sin()
    slope = (dC_dt * beta[:, None]).sum(-1)
    raw = slope[..., None] * X
    rhs = (Ct @ raw + torch.einsum("snj,sn,na->sja", dC_dt, residual, X)) / len(X)
    dbeta = -torch.linalg.solve(G, rhs)
    J = raw + C @ dbeta
    return prediction, beta, objective, J, dbeta


def optimize(X, y, theta, degree, steps, norm_bound, ridge=1e-12):
    damping = torch.full((len(theta),), 1e-3, dtype=X.dtype, device=X.device)
    eye = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    for _ in range(steps):
        pred, beta, objective, J, dbeta = profile(X, y, theta, degree, True, ridge)
        gram = J.transpose(1, 2) @ J / len(X) + ridge * dbeta.transpose(1, 2) @ dbeta
        rhs = (J * (pred - y)[..., None]).mean(1) + ridge * (dbeta * beta[..., None]).sum(1)
        scale = gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-10)
        delta = torch.linalg.solve(
            gram + (damping * scale)[:, None, None] * eye, rhs[..., None]
        ).squeeze(-1)
        trial = theta - delta
        trial *= (norm_bound / trial.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
        _, _, new_objective = profile(X, y, trial, degree, ridge=ridge)
        accepted = new_objective < objective
        theta = torch.where(accepted[:, None], trial, theta)
        damping = torch.where(accepted, damping * 0.4, damping * 5).clamp(1e-10, 1e10)
    return theta


def fit(
    public_path,
    family_path,
    starts=1024,
    steps=90,
    seed=91407,
    device="cpu",
    subset_rows=256,
    finalists=64,
    transfer_directions=16,
    range_screen=False,
):
    started = time.perf_counter()
    pub, data = load_public(public_path)
    meta, family = load_family(family_path)
    if (pub["D"], pub["Q"]) != (meta["D"], meta["Q"]):
        raise ValueError("Family differs from public specification")
    _, _, xn, yn = data["train"]
    _, _, xv, yv = data["validation"]
    torch.set_num_threads(2)
    X, y, V, v = [torch.tensor(a, device=device, dtype=torch.float64) for a in (xn, yn, xv, yv)]
    d, D = X.shape[1], pub["D"]
    screening = None
    keys = sorted(family)
    if range_screen:
        from ..screening import screen_candidates

        keys, screening = screen_candidates(pub, data, family)
        if not keys:
            return dict(
                method="variable-projection",
                status="UNRESOLVED",
                model=None,
                best_validation_mse=None,
                seconds=time.perf_counter() - started,
                public_sha256=sha(public_path),
                family_sha256=sha(family_path),
                screening=screening,
                uses_oracle=False,
            )
    u = (
        torch.quasirandom.SobolEngine(d + 1, scramble=True, seed=seed)
        .draw(starts)
        .to(device=device, dtype=X.dtype)
    )
    directions = torch.erfinv((2 * u[:, :d] - 1).clamp(-0.999999, 0.999999))
    directions /= directions.norm(dim=1, keepdim=True)
    directions *= torch.where(directions[:, :1] < 0, -1.0, 1.0)
    init = directions * (1 / D + (pub["norm_bound"] - 1 / D) * u[:, d:])
    eigenvalues, eigenvectors = np.linalg.eigh((xn * (yn - yn.mean())[:, None]).T @ xn / len(xn))
    spectral = (
        eigenvectors[:, np.argmax(abs(eigenvalues))][None]
        * np.linspace(1 / D, pub["norm_bound"], 9)[:, None]
    )
    init = torch.cat((init, torch.tensor(spectral, device=device, dtype=X.dtype)))
    nsub = min(subset_rows, len(X))
    with torch.no_grad():
        theta = optimize(X[:nsub], y[:nsub], init, D, steps, pub["norm_bound"])
        _, _, losses = profile(X, y, theta, D)
        theta = optimize(X, y, theta[losses.argsort()[:finalists]], D, 40, pub["norm_bound"])
        _, beta, losses = profile(X, y, theta, D)
        # Keep distinct directions in training-loss order. Validation is reserved
        # for choosing the final family member, never used for gradient updates.
        distinct = []
        for i in losses.argsort().cpu().tolist():
            t = theta[i].cpu().numpy()
            if all(min(np.linalg.norm(t - s), np.linalg.norm(t + s)) > 1e-5 for s in distinct):
                distinct.append(t)
            if len(distinct) >= transfer_directions:
                break
        ratios = sorted({a / b for a in range(1, D + 1) for b in range(1, D + 1)})
        seeds = [
            t * ratio
            for t in distinct
            for ratio in ratios
            if np.linalg.norm(t * ratio) <= pub["norm_bound"] + 1e-8
        ]
        init = torch.tensor(np.array(seeds), device=device, dtype=X.dtype)
        best = None
        attempts = []
        for k in keys:
            th = lm(X, y, init.clone(), k, pub["Q"], 40, pub["norm_bound"])
            vl = (response(th @ V.T, k, pub["Q"]) - v).square().mean(1)
            i = int(vl.argmin())
            model = dict(
                k=list(k), Q=pub["Q"], theta=[str(Fraction(float(t))) for t in th[i].cpu().numpy()]
            )
            attempts.append(dict(k=list(k), validation_mse=float(vl[i])))
            if best is None or float(vl[i]) < best[0]:
                best = (float(vl[i]), model)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return dict(
        method="variable-projection",
        status="FITTED",
        model=best[1],
        best_validation_mse=best[0],
        seconds=time.perf_counter() - started,
        starts=starts,
        spectral_starts=9,
        steps=steps,
        subset_rows=nsub,
        finalists=finalists,
        full_data_steps=40,
        transfer_steps=40,
        transfer_directions=len(distinct),
        transfer_seeds=len(seeds),
        harmonic_ratios=ratios,
        ridge=1e-12,
        device=device,
        dtype="float64",
        public_sha256=sha(public_path),
        family_sha256=sha(family_path),
        screening=screening,
        uses_oracle=False,
        attempts=attempts,
    )
