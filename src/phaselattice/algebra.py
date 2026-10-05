"""Exact output-band algebra and selected phase-list lattice decoding."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
import sympy as sp
from fpylll import LLL, IntegerMatrix

Fr = Fraction
_c = sp.Symbol("c")


def l1_sphere(D: int, Q: int) -> List[Tuple[int, ...]]:
    """All integer vectors k in Z^D with sum |k_j| = Q; beta = k/Q is B_{D,Q}."""
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            for s in (remaining, -remaining) if remaining else (0,):
                out.append(tuple(prefix + [s]))
            return
        for a in range(remaining + 1):
            for s in (a, -a) if a else (0,):
                rec(prefix + [s], remaining - a, slots - 1)

    rec([], Q, D)
    return sorted(set(out))


def normalize_period(k: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
    """Divide active indices by their gcd s; returns (normalized k, s)."""
    active = [j + 1 for j, kj in enumerate(k) if kj]
    if not active:
        raise ValueError("zero coefficient vector")
    s = 0
    for j in active:
        s = math.gcd(s, j)
    Dn = max(active) // s
    kn = [0] * Dn
    for j, kj in enumerate(k):
        if kj:
            kn[(j + 1) // s - 1] = kj
    return (tuple(kn), s)


def cheb_form(k: Sequence[int]) -> sp.Poly:
    """A(c) = sum_j k_j T_j(c) in Z[c]  (so that P_beta = A/Q)."""
    expr = sum((int(kj) * sp.chebyshevt(j + 1, _c) for j, kj in enumerate(k)))
    return sp.Poly(sp.expand(expr), _c)


@dataclass
class Band:
    """A certified output band of a period-normalized grid link."""

    k: Tuple[int, ...]
    Q: int
    chi: int
    comp: Tuple[Fraction, Fraction]
    J_plus: Tuple[Fraction, Fraction]
    J_zero: Tuple[Fraction, Fraction]
    J_minus: Tuple[Fraction, Fraction]
    c_intervals: List[Tuple[float, float]]
    Delta: Fraction
    rho: float
    alpha: float
    L: float
    min_h2: float
    unique_extremum: bool
    adjacent_to_extremum: bool
    status: str = "unverified"
    cert: Dict = field(default_factory=dict)

    def label(self) -> str:
        return f"k={self.k} Q={self.Q} chi={self.chi} J0=[{float(self.J_zero[0]):.4f},{float(self.J_zero[1]):.4f}]"


def _dyadic_inward(lo: Fraction, hi: Fraction, bits: int = 24) -> Tuple[Fraction, Fraction]:
    """Round [lo,hi] inward to 2^-bits Z."""
    s = 1 << bits
    a = Fr(math.ceil(lo * s), s)
    b = Fr(math.floor(hi * s), s)
    return (a, b)


def _mp_poly(A: sp.Poly):
    return [mp.mpf(int(x)) for x in A.all_coeffs()]


def _real_roots_in_unit(A: sp.Poly, y: mp.mpf, prec_bits: int) -> List[mp.mpf]:
    """Real roots of A(c) - y in [-1,1] at working precision, sorted increasing."""
    with mp.workprec(prec_bits):
        coeffs = _mp_poly(A)
        coeffs[-1] -= y
        roots = mp.polyroots(coeffs, maxsteps=400, extraprec=prec_bits)
        tol = mp.mpf(2) ** (-(prec_bits // 2))
        out = sorted(
            (r.real for r in roots if abs(r.imag) < tol and -1 - tol <= r.real <= 1 + tol),
            key=lambda v: float(v),
        )
        out = [min(max(r, mp.mpf(-1)), mp.mpf(1)) for r in out]
    return out


def _isolate(poly: sp.Poly, eps: sp.Rational) -> List[Tuple[sp.Rational, sp.Rational]]:
    """Disjoint rational isolating intervals (width <= eps) of the distinct real roots."""
    if poly.degree() < 1:
        return []
    sq = sp.Poly(sp.sqf_part(poly.as_expr()), poly.gens[0])
    if sq.degree() < 1:
        return []
    return [(sp.Rational(a), sp.Rational(b)) for (a, b), _ in sq.intervals(eps=eps)]


def _ival_eval(A: sp.Poly, a: sp.Rational, b: sp.Rational):
    """Interval evaluation of A on [a,b] (mpmath interval arithmetic at 200 bits)."""
    with mp.workprec(200):
        ia = mp.iv.mpf([str(a), str(b)])
        val = mp.iv.mpf(0)
        for co in A.all_coeffs():
            val = val * ia + mp.iv.mpf(int(co))
        return (val.a, val.b)


class SpecialValues:
    """Exact bookkeeping of the special values {A(1), A(-1)} u {critical values on (-1,1)}.

    Every special value receives a key: an integer index into the isolating intervals of
    the resultant R(y) = Res_c(A(c)-y, A'(c)) when it is a critical value, otherwise the
    rational itself.  Two special values are equal iff their keys agree, which is decided
    exactly (integer evaluation of R, isolating intervals of R and of A')."""

    def __init__(self, A: sp.Poly):
        self.A = A
        c = A.gens[0]
        y = sp.Symbol("y")
        Ad = A.diff(c)
        self.Ad = Ad
        R = (
            sp.Poly(sp.resultant(A.as_expr() - y, Ad.as_expr(), c), y)
            if Ad.degree() >= 1
            else sp.Poly(1, y)
        )
        self.R = R
        eps = sp.Rational(1, 2**20)
        self.r_int = _isolate(R, eps)
        cpts = _isolate(Ad, eps)
        self.crit = []
        for a, b in cpts:
            if a == b and abs(a) == 1:
                continue
            self.crit.append((a, b))
        self.crit_key = []
        while True:
            ok = True
            keys = []
            for a, b in self.crit:
                if b <= -1 or a >= 1:
                    keys.append(None)
                    continue
                if not (-1 < a and b < 1):
                    ok = False
                    break
                vlo, vhi = _ival_eval(A, a, b)
                hits = [
                    i
                    for i, (ra, rb) in enumerate(self.r_int)
                    if not (mp.mpf(str(rb)) < vlo or vhi < mp.mpf(str(ra)))
                ]
                if len(hits) != 1:
                    ok = False
                    break
                keys.append(hits[0])
            if ok:
                self.crit_key = keys
                break
            eps = eps / 2**8
            self.r_int = _isolate(R, eps)
            newc = _isolate(Ad, eps)
            self.crit = [(a, b) for a, b in newc if not (a == b and abs(a) == 1)]
        pairs = [(iv, k) for iv, k in zip(self.crit, self.crit_key) if k is not None]
        self.crit = [p[0] for p in pairs]
        self.crit_key = [p[1] for p in pairs]
        self.end_val = {"+": sp.Integer(A.eval(1)), "-": sp.Integer(A.eval(-1))}
        self.end_key = {}
        for s, v in self.end_val.items():
            if R.degree() >= 1 and R.eval(v) == 0:
                while True:
                    hits = [i for i, (ra, rb) in enumerate(self.r_int) if ra <= v <= rb]
                    if len(hits) == 1:
                        self.end_key[s] = hits[0]
                        break
                    eps = eps / 2**8
                    self.r_int = _isolate(R, eps)
                    self._recompute_crit_keys()
            else:
                while any((ra <= v <= rb for ra, rb in self.r_int)):
                    eps = eps / 2**8
                    self.r_int = _isolate(R, eps)
                    self._recompute_crit_keys()
                self.end_key[s] = ("rat", v)
        self.enclosure = {}
        for i, (ra, rb) in enumerate(self.r_int):
            self.enclosure[i] = (ra, rb)
        for s, v in self.end_val.items():
            k = self.end_key[s]
            if isinstance(k, tuple):
                self.enclosure[k] = (v, v)
        self.count: Dict = {}
        self.kind: Dict = {}
        for s in ("+", "-"):
            k = self.end_key[s]
            self.count[k] = self.count.get(k, 0) + 1
            self.kind[k] = self.kind.get(k, "") + f"end{s}"
        for k in self.crit_key:
            self.count[k] = self.count.get(k, 0) + 1
            self.kind[k] = self.kind.get(k, "") + "+crit"
        self.keys = sorted(self.count.keys(), key=lambda k: self.enclosure[k][0])
        for i in range(len(self.keys) - 1):
            if not self.enclosure[self.keys[i]][1] < self.enclosure[self.keys[i + 1]][0]:
                raise RuntimeError("special values not separated")

    def _recompute_crit_keys(self):
        keys = []
        for a, b in self.crit:
            vlo, vhi = _ival_eval(self.A, a, b)
            hits = [
                i
                for i, (ra, rb) in enumerate(self.r_int)
                if not (mp.mpf(str(rb)) < vlo or vhi < mp.mpf(str(ra)))
            ]
            assert len(hits) == 1
            keys.append(hits[0])
        self.crit_key = keys

    def values(self):
        return [
            (self.enclosure[k][0], self.enclosure[k][1], self.kind[k], self.count[k])
            for k in self.keys
        ]


def certify_link(
    k: Sequence[int], Q: int, chi_max: int = 2, bits: int = 24, certificates: bool = False
) -> Tuple[List[Band], Dict]:
    """Certify regular output bands with one or two folded inverse phases."""
    if chi_max not in (1, 2):
        raise ValueError("Only one/two-phase bands are supported")
    kn, s = normalize_period(k)
    A = cheb_form(kn)
    SV = SpecialValues(A)
    sv = SV.values()
    info = {
        "k_norm": kn,
        "period_gcd": s,
        "special_values": [(str(lo), str(hi), kind, cnt) for lo, hi, kind, cnt in sv],
        "components": [],
    }
    n_top = sv[-1][3]
    n_bot = sv[0][3]
    unique_ext = n_top == 1 or n_bot == 1
    info["max_attained_at"] = n_top
    info["min_attained_at"] = n_bot
    info["unique_extremum"] = unique_ext
    bands: List[Band] = []
    Dn = len(kn)
    Lc = 2 * math.pi * Dn
    for i in range(len(sv) - 1):
        lo_enc, hi_enc = (sv[i], sv[i + 1])
        comp_lo = sp.Rational(lo_enc[1])
        comp_hi = sp.Rational(hi_enc[0])
        if not comp_lo < comp_hi:
            continue
        y0 = (comp_lo + comp_hi) / 2
        fibers = sp.Poly(A.as_expr() - y0, _c).count_roots(-1, 1)
        adjacent = i == 0 or i == len(sv) - 2
        info["components"].append(
            {"lo": str(comp_lo), "hi": str(comp_hi), "fibers": int(fibers), "adjacent": adjacent}
        )
        if fibers == 0 or fibers > chi_max:
            continue
        chi = int(fibers)
        w = comp_hi - comp_lo
        Jp = _dyadic_inward(Fr(str(comp_lo + w / 4)), Fr(str(comp_hi - w / 4)), bits + 8)
        wp = Jp[1] - Jp[0]
        J0 = _dyadic_inward(Jp[0] + wp / 4, Jp[1] - wp / 4, bits)
        w0 = J0[1] - J0[0]
        Jm = _dyadic_inward(J0[0] + w0 / 4, J0[1] - w0 / 4, bits)
        if not Jm[0] < Jm[1]:
            continue
        Delta = min(J0[0] - Jp[0], Jp[1] - J0[1], Jm[0] - J0[0], J0[1] - Jm[1]) / Q
        prec = 120
        with mp.workprec(prec):
            r_lo = _real_roots_in_unit(A, mp.mpf(Jp[0].numerator) / Jp[0].denominator, prec)
            r_hi = _real_roots_in_unit(A, mp.mpf(Jp[1].numerator) / Jp[1].denominator, prec)
        if len(r_lo) != chi or len(r_hi) != chi:
            raise RuntimeError(f"fiber count mismatch on trimmed band for k={kn}")
        c_int = [(float(min(a, b)), float(max(a, b))) for a, b in zip(r_lo, r_hi)]
        Af = np.array([float(x) for x in A.all_coeffs()])
        Adf = np.polyder(Af)
        Addf = np.polyder(Adf)
        rho = math.inf
        alpha = 0.0
        min_h2 = math.inf
        with mp.workprec(prec):
            m_lo = _real_roots_in_unit(A, mp.mpf(Jm[0].numerator) / Jm[0].denominator, prec)
            m_hi = _real_roots_in_unit(A, mp.mpf(Jm[1].numerator) / Jm[1].denominator, prec)
        for (a, b), ma, mb in zip(c_int, m_lo, m_hi):
            cs = np.linspace(a, b, 2001)
            gp = (
                2 * math.pi * np.sqrt(np.clip(1 - cs**2, 0, None)) * np.abs(np.polyval(Adf, cs)) / Q
            )
            rho = min(rho, float(gp.min()))
            alpha += abs(float(mp.acos(ma) - mp.acos(mb))) / (2 * math.pi)
        if chi >= 2:
            for ell in range(chi):
                a, b = c_int[ell]
                cs = np.linspace(a + 1e-09, b - 1e-09, 401)
                for cc in cs:
                    yv = np.polyval(Af, cc)
                    roots = np.roots(Af - np.array([0] * (len(Af) - 1) + [yv]))
                    rr = sorted(
                        (r.real for r in roots if abs(r.imag) < 1e-09 and -1 <= r.real <= 1)
                    )
                    if len(rr) != chi:
                        continue
                    ell_idx = int(np.argmin([abs(r - cc) for r in rr]))
                    for j in range(chi):
                        if j == ell_idx:
                            continue
                        u = rr[j]
                        gp_t = -2 * math.pi * math.sqrt(1 - cc**2) * np.polyval(Adf, cc) / Q
                        gp_u = -2 * math.pi * math.sqrt(1 - u**2) * np.polyval(Adf, u) / Q
                        gpp_t = (
                            -((2 * math.pi) ** 2)
                            * (cc * np.polyval(Adf, cc) - (1 - cc**2) * np.polyval(Addf, cc))
                            / Q
                        )
                        gpp_u = (
                            -((2 * math.pi) ** 2)
                            * (u * np.polyval(Adf, u) - (1 - u**2) * np.polyval(Addf, u))
                            / Q
                        )
                        hp = gp_t / gp_u
                        hpp = (gpp_t - gpp_u * hp**2) / gp_u
                        min_h2 = min(min_h2, abs(hpp))
        bands.append(
            Band(
                k=kn,
                Q=Q,
                chi=chi,
                comp=(Fr(str(comp_lo)) / Q, Fr(str(comp_hi)) / Q),
                J_plus=(Jp[0] / Q, Jp[1] / Q),
                J_zero=(J0[0] / Q, J0[1] / Q),
                J_minus=(Jm[0] / Q, Jm[1] / Q),
                c_intervals=c_int,
                Delta=Delta,
                rho=rho,
                alpha=alpha,
                L=Lc,
                min_h2=min_h2 if chi >= 2 else math.nan,
                unique_extremum=unique_ext,
                adjacent_to_extremum=adjacent,
            )
        )
    for b in bands:
        if b.chi <= 2:
            b.status, b.cert = ("exact:chi<=2", {"rule": "definition"})
    return (bands, info)


def round_dyadic(v, B: int) -> Fraction:
    """Nearest-dyadic rounding to 2^-B Z (ties away from zero), exact output."""
    scaled = mp.mpf(v) * (1 << B)
    r = mp.floor(scaled + mp.mpf("0.5")) if scaled >= 0 else mp.ceil(scaled - mp.mpf("0.5"))
    return Fr(int(r), 1 << B)


def link_value(kn: Sequence[int], Q: int, t) -> mp.mpf:
    """g(t) = sum_j (k_j/Q) cos(2 pi j t) at the working precision."""
    s = mp.mpf(0)
    for j, kj in enumerate(kn):
        if kj:
            s += mp.mpf(kj) * mp.cos(2 * mp.pi * (j + 1) * t)
    return s / Q


def phase_list(band: Band, ybar: Fraction, B: int) -> List[Fraction]:
    """All folded inverse phases of a rounded label on the band, rounded to B bits."""
    A = cheb_form(band.k)
    prec = B + 64
    with mp.workprec(prec):
        y = mp.mpf(ybar.numerator) / ybar.denominator * band.Q
        roots = _real_roots_in_unit(A, y, prec)
        sel = []
        for a, b in band.c_intervals:
            cands = [r for r in roots if a - 1e-09 <= float(r) <= b + 1e-09]
            if len(cands) != 1:
                cands = [min(roots, key=lambda r: abs(float(r) - 0.5 * (a + b)))]
            sel.append(cands[0])
        return [round_dyadic(mp.acos(r) / (2 * mp.pi), B) for r in sel]


def _rational_inverse(G: List[List[int]]) -> Tuple[int, List[List[Fraction]]]:
    """Invert an integer matrix by exact rational Gauss-Jordan elimination."""
    n = len(G)
    M = [[Fr(G[i][j]) for j in range(n)] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return (0, [])
        M[col], M[piv] = (M[piv], M[col])
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return (1, [row[n:] for row in M])


@dataclass
class DecodeResult:
    status: str
    theta_hat: Optional[np.ndarray] = None
    branches: Optional[List[int]] = None
    signs: Optional[List[int]] = None
    lifts: Optional[List[int]] = None
    multiple: Optional[int] = None
    reduction_seconds: float = 0.0
    short_norm: float = math.nan
    spurious: Optional[Tuple[List[int], List[int]]] = None
    theta_exact: Optional[List[Fraction]] = None


def decode_mr(
    Xbar: List[List[Fraction]],
    lists: List[List[Fraction]],
    chi: int,
    B: int,
    Gamma: float,
) -> DecodeResult:
    """Decode phase branches, signs and integer lifts using rational projection and LLL."""
    m = len(Xbar)
    d = len(Xbar[0])
    N = B
    scale = 1 << B
    Xi = [[int(x * scale) for x in row] for row in Xbar]
    norms = [math.sqrt(sum(((v / scale) ** 2 for v in row))) for row in Xi]
    Rx = max(1.0, max(norms))
    Kmax = int(math.ceil(Gamma * Rx)) + 2
    G = [[sum((Xi[i][a] * Xi[i][b] for i in range(m))) for b in range(d)] for a in range(d)]
    ok, Ginv = _rational_inverse(G)
    if not ok:
        return DecodeResult("FAIL-rank")
    XG = [[sum((Fr(Xi[i][a]) * Ginv[a][b] for a in range(d))) for b in range(d)] for i in range(m)]
    P = [
        [Fr(int(i == l)) - sum((XG[l][b] * Xi[i][b] for b in range(d))) for i in range(m)]
        for l in range(m)
    ]
    twoN = 1 << N
    Pk = [[math.floor(P[l][i] * twoN) for i in range(m)] for l in range(m)]
    Pa = [
        [math.floor(P[l][i] * lists[i][j] * twoN) for i in range(m) for j in range(chi)]
        for l in range(m)
    ]
    nprime = (2 + chi) * m
    U = 3 * m**1.5 * (Kmax + 2)
    Lam = 2.0 ** ((nprime - 1) / 2)
    H = Lam * U
    M = math.ceil(H) + 1
    rows = []
    for i in range(m):
        rows.append([M * Pk[l][i] for l in range(m)] + [int(i == t) for t in range(nprime)])
    for col in range(m * chi):
        rows.append([M * Pa[l][col] for l in range(m)] + [int(m + col == t) for t in range(nprime)])
    for l0 in range(m):
        rows.append(
            [M * int(l0 == l) for l in range(m)]
            + [int(m + m * chi + l0 == t) for t in range(nprime)]
        )
    t0 = time.time()
    A = IntegerMatrix.from_matrix(rows)
    LLL.reduction(A)
    red_s = time.time() - t0
    red = [[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)]
    red.sort(key=lambda v: sum((x * x for x in v)))
    short = red[0]
    short_norm = math.sqrt(sum((x * x for x in short[m:])))
    if any(short[:m]):
        return DecodeResult("FAIL-residual", reduction_seconds=red_s, short_norm=short_norm)
    if short_norm > H:
        return DecodeResult("FAIL-long", reduction_seconds=red_s, short_norm=short_norm)
    t = short[m:]
    tk, tphi = t[:m], t[m : m + m * chi]
    if not any(tk) and (not any(tphi)):
        return DecodeResult("FAIL-zero", reduction_seconds=red_s, short_norm=short_norm)
    g0 = 0
    for v in tphi:
        g0 = math.gcd(g0, abs(v))
    if g0 == 0:
        return DecodeResult(
            "FAIL-gcd0", reduction_seconds=red_s, short_norm=short_norm, spurious=(tk, tphi)
        )
    if any((v % g0 for v in tk)) or any((v % g0 for v in tphi)):
        return DecodeResult(
            "FAIL-div", reduction_seconds=red_s, short_norm=short_norm, spurious=(tk, tphi)
        )
    k_hat = [v // g0 for v in tk]
    eps_hat, b_hat = ([], [])
    for i in range(m):
        blk = [tphi[chi * i + j] // g0 for j in range(chi)]
        supp = [j for j, v in enumerate(blk) if v]
        if len(supp) != 1 or abs(blk[supp[0]]) != 1:
            return DecodeResult(
                "FAIL-onehot", reduction_seconds=red_s, short_norm=short_norm, spurious=(tk, tphi)
            )
        b_hat.append(supp[0])
        eps_hat.append(blk[supp[0]])
    if any((abs(v) > Kmax for v in k_hat)):
        return DecodeResult(
            "FAIL-lift", reduction_seconds=red_s, short_norm=short_norm, spurious=(tk, tphi)
        )
    z = [Fr(eps_hat[i]) * lists[i][b_hat[i]] + k_hat[i] for i in range(m)]
    rhs = [sum((Fr(Xi[i][a]) * z[i] for i in range(m))) for a in range(d)]
    th = [sum((Ginv[a][b] * rhs[b] for b in range(d))) * scale for a in range(d)]
    theta_hat = np.array([float(v) for v in th])
    if np.linalg.norm(theta_hat) > 2 * Gamma:
        return DecodeResult("FAIL-norm", reduction_seconds=red_s, short_norm=short_norm)
    resid = max(
        (abs(float(sum((Fr(Xi[i][a]) * th[a] for a in range(d))) / scale - z[i])) for i in range(m))
    )
    if resid > 2.0 ** (-N / 2):
        return DecodeResult("FAIL-resid", reduction_seconds=red_s, short_norm=short_norm)
    return DecodeResult(
        "OK",
        theta_hat=theta_hat,
        branches=b_hat,
        signs=eps_hat,
        lifts=k_hat,
        multiple=g0,
        reduction_seconds=red_s,
        short_norm=short_norm,
        theta_exact=th,
    )


def select_and_invert(band: Band, xs, ys, m: int, B: int):
    """Retain the first m observations whose rounded label lies in J_0; invert lists."""
    Xr, Lr, idx = ([], [], [])
    for i, (x, y) in enumerate(zip(xs, ys)):
        if band.J_zero[0] <= y <= band.J_zero[1]:
            Xr.append(x)
            Lr.append(phase_list(band, y, B))
            idx.append(i)
            if len(Xr) == m:
                break
    return (Xr, Lr, idx)


def predict(kn, Q, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    t = X @ theta
    out = np.zeros(len(X))
    for j, kj in enumerate(kn):
        if kj:
            out += kj * np.cos(2 * np.pi * (j + 1) * t)
    return out / Q


def _clip1(v):
    return mp.mpf(-1) if v < -1 else mp.mpf(1) if v > 1 else v


def quantized_validation_loss(kn, Q, theta_exact, xs_val, ys_val, prec: int) -> float:
    """Empirical squared loss of the candidate (link, exact theta) on the rounded validation
    sample, as in Algorithm 1: the prediction is evaluated at the working precision from the
    exact rational theta, and prediction and observed label are clipped to [-1,1]."""
    with mp.workprec(prec):
        th = [mp.mpf(v.numerator) / v.denominator for v in theta_exact]
        acc = mp.mpf(0)
        for x, y in zip(xs_val, ys_val):
            t = mp.fsum((mp.mpf(xa.numerator) / xa.denominator * ta for xa, ta in zip(x, th)))
            pv = _clip1(link_value(kn, Q, t))
            yv = _clip1(mp.mpf(y.numerator) / y.denominator)
            acc += (pv - yv) ** 2
        return float(acc / len(xs_val))


def certified_family(
    D: int, Q: int, chi_max: int = 2, verbose: bool = False, certificates: bool = False
):
    """Enumerate B_{D,Q}, normalize, certify; returns (bands_by_link, census)."""
    fam: Dict[Tuple[int, ...], List[Band]] = {}
    census = {
        "grid": 0,
        "distinct_normalized": 0,
        "covered": 0,
        "unique_extremum": 0,
        "fiber_hist": {},
    }
    seen = set()
    for k in l1_sphere(D, Q):
        census["grid"] += 1
        kn, s = normalize_period(k)
        if kn in seen:
            continue
        seen.add(kn)
        census["distinct_normalized"] += 1
        bands, info = certify_link(kn, Q, chi_max=chi_max, certificates=certificates)
        if info["unique_extremum"]:
            census["unique_extremum"] += 1
        for comp in info["components"]:
            census["fiber_hist"][comp["fibers"]] = census["fiber_hist"].get(comp["fibers"], 0) + 1
        if bands:
            census["covered"] += 1
            fam[kn] = bands
        if verbose:
            print(kn, "bands:", [(b.chi, float(b.J_zero[0]), float(b.J_zero[1])) for b in bands])
    return (fam, census)
