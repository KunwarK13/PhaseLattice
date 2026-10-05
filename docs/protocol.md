# Evaluation protocol

The benchmark models are fixed in [cases.json](../evidence/cases.json). Each
data-generating model is called a teacher. All methods receive the same public
data. Inference does not read reference parameters, seeds, or test labels.

## Cohorts

| Cohort | Dimensions | Models per dimension | Direction norm | Public upper bound |
|---|---|---:|---|---:|
| Analytic | 3, 6, 10 | 8 | 1.5 | 2 |
| Qiskit | 4, 8 | 4 | 1.5 | 2 |
| Extension | 4, 8, 12 | 4 | 1.1, 1.5, 1.9, 2.5 | 3 |

The analytic and Qiskit cohorts use seeds `510000 + 100*d + i` and
`610000 + 100*d + i`. Extension seeds are `940000 + 100*d + i`; at each dimension,
the `i=1` teacher uses Qiskit and the other three are analytic. There are 44
independent hidden directions in total, including 11 Qiskit teachers.

Each model chooses a waveform uniformly from the complete period-normalized
`D=3, Q=4` grid of 62 waveforms. It supplies 1,024 training observations, 128 validation
observations, and 2,048 evaluation observations. The public observations have
40 fractional bits. Gaussian acquisition and response evaluation use guard
precision before rounding. Qiskit expectations use finite-precision statevector
arithmetic and are checked against the analytic equation during generation.

## Recovery criterion

All three conditions must hold:

1. The normalized integer waveform coefficients and denominator are correct.
2. Weight error, allowing the global sign symmetry, is below `1e-4`.
3. Independent test MSE divided by target variance is below `1e-6`.

Prediction success is recorded separately. An unresolved reconstruction is a
failure, not an excluded trial. A fitter always returns its best model even when
it fails the criterion. No algorithm is initialized with the hidden parameters.

## Optimization controls

| Method | Exploration | Full-data refinement |
|---|---|---|
| Multistart LM | Every waveform; 256 Sobol starts and five spectral starts; 60 steps on 128 rows | 12 finalists per waveform; 30 steps |
| Inverse-phase projection + LM | Every waveform; all inverse branches; two annealing schedules; same LM budget | Same finalist refinement |
| Harmonic transfer | Reuse the two fitted directions across waveforms and small harmonic ratios | 40 steps on all training rows |
| Variable projection | Solve three real coefficients per direction; 1,024 Sobol and nine spectral starts; exact profiled Jacobian; 90 steps on 256 rows | 64 finalists, 40 steps; 16 distinct directions transfer across harmonics and waveforms for another 40 steps |

All controls use float64 and can represent every benchmark teacher. Variable
projection uses ridge `1e-12` in its profiled linear solve. Exact range filtering
is supplied to its final discrete-family search. The optimizer portfolio chooses
between harmonic transfer and variable projection by validation MSE, counting
the work of both methods. It never selects by test performance.

The comparison is limited to these optimization budgets. The controls include
successful low-dimensional cases, a finite-difference check of the profiled
Jacobian, and recovery of a continuous
waveform from a nearby starting direction.

## Derivatives and quantum response

Every reconstructed model is tested on 512 inputs from each of four distributions:
standard Gaussian, Gaussian scaled by three, Gaussian translated by two in every
coordinate, and independent uniform `[-3,3]`. Input gradients and Hessian-vector
products are compared with independent PyTorch differentiation of the reference
teacher. The threshold is relative RMS error below `1e-6` for every distribution.

Qiskit reconstructions are compared with original circuit expectations on eight
translated inputs per teacher. Five-point finite differences, with step `1e-4`,
independently check gradients of the original circuit. These are additional
evaluations of the same teachers, not additional reconstruction trials.

## Timing

Inference timing includes loading public data, hypothesis search, and validation.
It excludes teacher generation, test scoring, source capture, process/import
startup, and one-time family compilation. CUDA initialization is warmed before
timing. CPU/GPU timings are descriptive measurements on the recorded workstation.

The range-filter comparison uses the first two original analytic teachers at
dimensions 3, 6, and 10. Full and screened runs alternate order between paired
teachers; range caches are cleared before every run. Both must return identical
models with valid traces. Speedup is measured against the same decoder without
filtering.

## Negative controls and precision

The same six analytic teachers are evaluated at 12, 20, 28, 32, 36, and 40
fractional bits by coarsening their saved observations. These are correlated
precision measurements on six teachers. Single-relation and independently
shuffled-label controls use the same cases.

The shot-noise control uses three additional Qiskit teachers at 1,024 and 16,384
shots per observation, with 256 training and 128 validation observations.
Filtering is disabled for these noisy observations. Approximate prediction is
defined separately as test MSE/variance below `1e-3`; the stricter reconstruction
criterion is unchanged. There are three independent teachers, not six.

## Provenance

The benchmark data were copied from the development evaluation without
changing any numerical observation. [Migration records](../evidence/migration.json)
retain original byte digests, new file digests, and canonical numerical digests.
Schema identifiers, compression, and evaluator filenames changed.

The package was evaluated again on these fixed observations.
Each run records package versions, hardware, source hashes, a source snapshot,
family and cohort hashes, and every per-case result. Resumption rejects a changed
source, configuration, environment, or observation digest. The original expected
equations are retained for regression checks after inference, never as inputs.
