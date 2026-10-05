"""Transfer fitted directions across waveform candidates and harmonic ratios."""

import time
from fractions import Fraction

import numpy as np
import torch

from ..family import load_family
from ..io import sha256 as sha
from ..observations import load_public
from .least_squares import lm, response


def refine(public, family, initial_results, device="cpu"):
    t0 = time.perf_counter()
    pub, data = load_public(public)
    _, fam = load_family(family)
    _, _, xn, yn = data["train"]
    _, _, xv, yv = data["validation"]
    torch.set_num_threads(2)
    X = torch.tensor(xn, dtype=torch.float64, device=device)
    y = torch.tensor(yn, dtype=torch.float64, device=device)
    V = torch.tensor(xv, dtype=torch.float64, device=device)
    v = torch.tensor(yv, dtype=torch.float64, device=device)
    ratios = sorted({a / b for a in range(1, pub["D"] + 1) for b in range(1, pub["D"] + 1)})
    seeds = []
    for result in initial_results:
        th = np.array([float(Fraction(z)) for z in result["model"]["theta"]])
        for ratio in ratios:
            candidate = th * ratio
            if np.linalg.norm(candidate) <= pub["norm_bound"] + 1e-08:
                seeds.append(candidate)
    init = torch.tensor(np.array(seeds), dtype=torch.float64, device=device)
    best = None
    with torch.no_grad():
        for k in sorted(fam):
            theta = lm(X, y, init.clone(), k, pub["Q"], 40, pub["norm_bound"])
            losses = (response(theta @ V.T, k, pub["Q"]) - v).square().mean(1)
            index = int(losses.argmin())
            loss = float(losses[index])
            if best is None or loss < best[0]:
                model = dict(
                    k=list(k),
                    Q=pub["Q"],
                    theta=[str(Fraction(float(t))) for t in theta[index].cpu().numpy()],
                )
                best = (loss, model)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return dict(
        method="optimizer-transfer",
        status="FITTED",
        model=best[1],
        best_validation_mse=best[0],
        seconds=seconds + sum((r["seconds"] for r in initial_results)),
        additional_seconds=seconds,
        seeds=len(seeds),
        harmonic_ratios=ratios,
        device=device,
        public_sha256=sha(public),
        initial_methods=[r["method"] for r in initial_results],
        uses_oracle=False,
    )
