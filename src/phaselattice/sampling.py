"""Gaussian acquisition with guard precision before dyadic rounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mpmath as mp
import numpy as np

from .algebra import link_value, round_dyadic
from .algebra import predict as predict


@dataclass
class Instance:
    d: int
    B: int
    gamma: float
    theta: List[mp.mpf]
    theta_f: np.ndarray


def mp_gauss(rng: np.random.Generator, n: int, prec: int) -> List[mp.mpf]:
    """Generate n Gaussian samples using high-precision Box-Muller transforms.

    Each pair uses two uniforms drawn from rng.bytes, with prec rounded up to a
    whole number of bytes. The transforms use prec + 64 bits of working precision.
    Samples remain mpmath values; no float64 conversion is used.
    """
    if n <= 0:
        return []
    half = (n + 1) // 2
    nbytes = (prec + 7) // 8
    buf = rng.bytes(2 * half * nbytes)
    out: List[mp.mpf] = []
    with mp.workprec(prec + 64):
        scale = mp.mpf(2) ** (-8 * nbytes)
        two_pi = 2 * mp.pi
        for i in range(half):
            o = 2 * i * nbytes
            k1 = int.from_bytes(buf[o : o + nbytes], "big")
            k2 = int.from_bytes(buf[o + nbytes : o + 2 * nbytes], "big")
            u1 = (mp.mpf(k1) + 1) * scale
            u2 = mp.mpf(k2) * scale
            radius = mp.sqrt(-2 * mp.log(u1))
            angle = two_pi * u2
            out.append(radius * mp.cos(angle))
            if len(out) < n:
                out.append(radius * mp.sin(angle))
    return out


def make_direction(d: int, gamma: float, rng: np.random.Generator, prec: int) -> Instance:
    """A latent direction of norm gamma, drawn and normalized at the working precision."""
    with mp.workprec(prec + 64):
        v = mp_gauss(rng, d, prec)
        sc = mp.mpf(gamma) / mp.sqrt(mp.fsum((a * a for a in v)))
        th = [a * sc for a in v]
    return Instance(d=d, B=0, gamma=gamma, theta=th, theta_f=np.array([float(a) for a in th]))


def draw_observations(inst, kn, Q, n: int, B: int, rng, perturb=None, noise_sigma: float = 0.0):
    """n rounded observations (xbar, ybar) plus the latent projections (for diagnostics).

    perturb(t) -> mp value: a projection-measurable pre-rounding perturbation xi (bounded);
    noise_sigma: independent Gaussian label noise added before rounding."""
    prec = B + 64
    d = inst.d
    xs, ys, ws = ([], [], [])
    with mp.workprec(prec):
        for _ in range(n):
            x = mp_gauss(rng, d, prec)
            w = mp.fsum((a * b for a, b in zip(inst.theta, x)))
            y = link_value(kn, Q, w)
            if perturb is not None:
                y = y + perturb(w)
            if noise_sigma > 0:
                y = y + mp_gauss(rng, 1, prec)[0] * mp.mpf(noise_sigma)
            xs.append([round_dyadic(xi, B) for xi in x])
            ys.append(round_dyadic(y, B))
            ws.append(w)
    return (xs, ys, ws)
