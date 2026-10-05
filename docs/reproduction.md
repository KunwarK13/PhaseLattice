# Reproduction

Run these commands from the repository root. The package was evaluated on Linux
with Python 3.11. The CI workflow is configured to test the core package on Python
3.11 and 3.12. Qiskit runs locally without an IBM account or API token.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m pip install --no-deps -e .
```

The requirements pin the direct numerical and development dependencies.
Transitive packages and platform-specific binary builds are not fully locked.
The recorded GPU run used PyTorch `2.5.1+cu121`; install that build before the
requirements when reproducing the same CUDA environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only work, use the same command with the `/whl/cpu` index. The plain
`torch==2.5.1` requirement accepts either local build. The core decoder needs
neither PyTorch nor Qiskit:

```bash
python -m pip install -e .
phaselattice recover evidence/cases/analytic_d03_s01/observations.json.gz \
  --output runs/reconstruction.json
phaselattice verify evidence/cases/analytic_d03_s01/observations.json.gz \
  runs/reconstruction.json
```

## Demonstration and notebook

```bash
phaselattice demo --output runs/demo
python examples/reconstruct.py
python examples/execute_notebook.py
```

The demo generates a new Qiskit teacher and runs inference in a separate process
that receives only its public observations and output paths. The result includes
a trace, a standalone Python model, a `.qpy` circuit, and derivative evaluations.
An existing demo directory containing JSON records is rejected to avoid mixing
experiments; select a new directory to run another trial.

The [executed notebook](../examples/reconstruction.ipynb) uses one frozen Qiskit
case and walks through the same public interface. Its evaluation reads the
hidden reference only after reconstruction. The execution helper uses the
current Python interpreter and removes timing and machine-path metadata from
the notebook.

## Tests and package build

```bash
pytest
ruff check .
ruff format --check .
python -m build
```

Tests cover the observation boundary, inverse-phase assignments, trace tampering,
exact range filtering, derivatives, parameter ordering in twelve-feature quantum
circuits, finite-shot independence, and optimizer positive controls. Tests for
optional integrations are skipped when those dependencies are absent.

The wheel contains the numerical package, the default compiled family, and the
standalone export template. The benchmark data, figures and notebooks are
repository artifacts rather than package dependencies.

## Full comparison

```bash
python benchmarks/run.py --device cuda --output runs/benchmark
python benchmarks/figures.py --run runs/benchmark --output runs/figures \
  --results runs/results.md
python benchmarks/audit.py --run runs/benchmark --figures runs/figures --output runs/audit.json
```

Use `--device cpu` if CUDA is unavailable. The lattice decoder always runs on CPU.
The experiment includes 44 models, six methods, derivative evaluations, six paired timings, a
precision sweep, ablations and shot-noise controls. See the
[protocol](protocol.md) for budgets and thresholds.

Individual stages are available as `--stage recovery`, `optimizers`,
`derivatives`, `timing`, and `controls`. Recovery and optimizer results must
exist before the derivatives stage. `--limit N` supports a short exploratory
run; it is not a full benchmark and cannot produce the results page.

The runner resumes existing per-case results only when source, protocol, cohort,
environment, device and observation digests agree. If any of those change, use
a fresh output directory. A result file is never silently reused for different
observations. Running one stage refreshes the relevant summary files.

## Inspect the recorded results

```bash
python benchmarks/figures.py
python benchmarks/audit.py
```

These commands use `evidence/run`, generate SVG/PDF/PNG figures from its numerical
records, and verify the stored data and results. The audit recomputes held-out
scores and trace checks. It checks derivative records for consistency; rerun
the derivatives stage to regenerate the independent autograd and circuit
measurements themselves.

Timings depend on hardware, process load, library builds and caching. Numerical
reconstruction comparisons use the saved rational models and prescribed error
thresholds. Optimizer trajectories may differ across platforms.

## Reconstruction animation

```bash
python benchmarks/animate.py
```

This renders the eight-second README animation as a GIF and a 1080p H.264 video,
plus a static SVG and PNG. Matplotlib and an `ffmpeg` executable on `PATH` are
required; the recorded render uses FFmpeg 6.1.1. `--poster-only` renders the
static alternatives without FFmpeg.

The animation uses the saved `quantum_d04_s03` observations and a verified
reconstruction trace. It displays 256 of the 1,024 training observations,
including all eight selected rows. The right panel shows phase choices for
the recovered candidate waveform within the displayed cycle window. The
consistent assignments are highlighted together. The left panel then changes
from the first input coordinate to the inferred projection modulo one.
Indigo marks observed responses, coral identifies the selected observations
and their phase assignments, and teal traces the recovered equation. The
sequence ends with the recovered waveform; the GIF loops directly to its
opening frame without a reverse transition.

The point motion illustrates a coordinate change. It does not represent
optimizer iterations, intermediate LLL states, or measured inference time.
The animation does not show the search across candidate waveforms. Reference
parameters are not read by the renderer.

The [media manifest](media/manifest.json) records the displayed rows, chapter
times, renderer and data hashes, encoding settings, and artifact hashes.
The evidence audit checks those hashes alongside the scientific figures.
Use the [static view](media/reconstruction.svg) to inspect the result without
motion, or the [video](media/reconstruction.mp4) for playback controls.

## Regenerate acquisition or the family

```bash
phaselattice generate --output runs/new-teacher --dimension 6 --seed 510600
phaselattice family --output runs/family.json
```

Generation parameters and seeds for the frozen cohort are listed in
[cases.json](../evidence/cases.json). The original observations remain the
authoritative benchmark input; a new Qiskit version can change statevector
rounding. Recompiling the family should preserve its canonical band-content
digest. Compilation duration is metadata, so the full file digest can change.
Never replace an observation or family file inside an existing measured run.
