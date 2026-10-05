# Selected phase-list reconstruction

PhaseLattice solves a structured inverse problem. The observations are generated
by a linear feature followed by an unknown periodic waveform:

$$y=g_k(\theta^\top x),\qquad
g_k(t)=\sum_{j=1}^{D}\frac{k_j}{Q}\cos(2\pi jt),\qquad
\sum_j|k_j|=Q.$$

The learner knows the finite coefficient family, an upper bound on the weight
norm, and the observation precision. It receives neither the chosen coefficient
vector nor the hidden weights. Inputs are drawn from a standard Gaussian and
both inputs and outputs are rounded to a dyadic grid.

The implementation uses period-normalized coefficient vectors. A common divisor
of the active harmonic indices is absorbed into the hidden direction. Since
every waveform is even, weights are identifiable only up to a global sign.

## Inverse-phase lists

Write `c = cos(2πt)`. Each candidate waveform becomes a low-degree polynomial in
`c` through the Chebyshev identity `cos(2πjt) = T_j(c)`. Critical values divide the
output range into intervals with a constant number of inverse branches.

The family compiler uses polynomial root isolation and exact root counts to
construct trimmed regular bands. The implementation uses bands with one or two
folded inverse phases. Numerical estimates of branch width and curvature are
recorded, but they do not certify nonresonance.

For an observed output inside a band, inversion returns a finite list of phases
in `[0, 1/2]`. The unknown projection satisfies

$$\theta^\top x_i=n_i+\varepsilon_i a_{i,b_i},
\qquad n_i\in\mathbb Z,\quad\varepsilon_i\in\{-1,+1\}.$$

The unknowns include a branch, a reflection, and an integer winding number for
each selected observation. Solving those choices independently discards the
constraint that they all arise from one weight vector.

## Joint phase assignment using linear relations

Stack the selected inputs into `X`, with `m` rows and `d` columns. If `z` contains
the corresponding unwrapped projections, then `z = Xθ`. For full-rank `X`,

$$P=I-X(X^\top X)^{-1}X^\top,\qquad Pz=0.$$

`P` projects onto the orthogonal complement of the column space of `X`, equivalently
the kernel of `Xᵀ`. The implementation computes this projector with rational
arithmetic from the observed dyadic inputs. It substitutes each candidate phase
list into the relations and encodes integer windings, signed branch indicators,
and rounding-completion coordinates in an integer lattice.

LLL reduction returns a short relation. The decoder checks the lattice residual,
divisibility, signed one-hot branch pattern, and winding bounds. The
selected assignments determine an exact rational least-squares weight estimate.
Held-out observations choose among the surviving waveform hypotheses.

The default uses `m = 2d` selected rows. These rows are drawn from the full
training pool; **2d is not the total observation cost**. The benchmark uses 1,024
training observations and another 128 validation observations for every method.

The lattice encodes the shared phase relations. Recovery depends on observation
precision and the selected regular output bands. The decoder is designed for
this structure; it is not a solver for general hard lattice problems.

## Exact range filtering

Before inversion, a candidate can be rejected when its entire output range is
incompatible with an observed label. All extrema of the Chebyshev polynomial on
`[-1,1]` occur at endpoints or derivative roots. Rational root isolation and
interval Horner evaluation enclose these extrema.

Filtering uses training labels only and allows one dyadic unit for rounding.
It is conservative under the precise-response model: no label within a candidate's
true range is grounds for rejection. The optimization control receives the same
filter in its final discrete-family search. Filtering is disabled for shot-noisy
observations because they violate the range assumption.

## Trace verification

The exported trace contains selected row indices, inverse phase lists, chosen
branches, signs, and winding numbers. A separate verifier checks:

1. The observation digest and public model specification.
2. Label membership in the selected band and inverse-response consistency.
3. Full column rank of the selected input matrix.
4. Exact rational least-squares normal equations and a small phase residual.
5. Agreement with independent validation observations.

These checks establish a consistent reconstruction trace. They do not establish
uniqueness among every possible model. The `1e-8` validation-MSE acceptance rule
is a diagnostic threshold, not a statistical confidence certificate.

## Input gradients and Hessian-vector products

For `f(x) = g(θᵀx)`, the recovered model provides

$$\nabla_x f(x)=g'(\theta^\top x)\theta,\qquad
\nabla_x^2 f(x)v=g''(\theta^\top x)(\theta^\top v)\theta.$$

No derivative observations are used during learning. The evaluation compares
these derivatives with independent PyTorch automatic differentiation of the
data-generating model (the teacher) on four input distributions.

## Quantum realization and reconstruction

A selector register is prepared with amplitudes proportional to the square roots
of the absolute harmonic coefficients. Conditional rotations apply harmonic
multiples of the projected input angle to a response qubit. Conditional sign
operations implement negative coefficients. The response-qubit `Z` expectation
equals the specified cosine sum in ideal arithmetic.

The learner sees only this scalar expectation. After reconstruction, the same
circuit construction applied to the inferred equation yields a circuit with the
recovered response. It does not recover a unique original gate sequence, full
quantum state, or responses to unobserved measurement operators.

The benchmark circuits use at most three qubits and are classically simulable.
They test agreement between the measured circuit response and the reconstructed
equation. See [research context](references.md) for related work on lattice-based
learning and quantum model extraction.
