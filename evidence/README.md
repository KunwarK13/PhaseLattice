# Evidence index

This directory contains the benchmark data and the package's recorded results.
Each teacher is a synthetic model that generates observations. Reference
parameters are evaluation data and are excluded from the inference interface.

| Artifact | Contents |
|---|---|
| [cases.json](cases.json) | 44 independent teachers, generation settings, observation/reference/test hashes |
| [cases/](cases) | Compressed public observations, hidden reference parameters, test arrays, expected regression models |
| [migration.json](migration.json) | Original archive fingerprint and canonical numerical digests retained during migration |
| [run/manifest.json](run/manifest.json) | Source fingerprints, versions, platform, device and cohort identity |
| [run/source.tar.gz](run/source.tar.gz) | Source and protocol snapshot used in the benchmark |
| [source_documentation.json](source_documentation.json) | Hashes of later comment and docstring edits; checked against the archived Python code |
| [run/summary.json](run/summary.json) | Six methods × 44 teachers; recovery, prediction, error and timing records |
| [run/derivatives.json](run/derivatives.json) | Derivative checks across four distributions and 11 circuit round-trips |
| [run/timing.json](run/timing.json) | Six paired full-family/range-filtered timings |
| [run/precision.json](run/precision.json) | The same six teachers at six precision settings |
| [run/ablations.json](run/ablations.json) | Single-relation decoding and shuffled-label controls |
| [run/noise.json](run/noise.json) | Three additional teachers, each at two shot budgets |
| [audit.json](audit.json) | Recomputed scores, trace replay and integrity checks |
| [validation.json](validation.json) | Tests, clean wheel installation, executed notebook and Qiskit demo |

Each `run/<case>/` directory retains its inference and optimization records.
The reconstruction record includes selected rows, inverse-phase lists, signs,
integer lifts, rational weights, validation loss, and all attempted hypotheses.
The decoder reads only `observations.json.gz` and the compiled public family.
`reference.json`, `test.npz`, and `expected_model.json` are opened after inference
for evaluation or regression checks.

The original run manifest and source archive are unchanged. Later source edits
are limited to comments and docstrings and are listed in `source_documentation.json`.
The audit checks their hashes and compares the parsed Python code with the archive
after removing docstrings. Changes to executable statements fail this check.

The 44 high-precision teachers include 11 Qiskit circuits. The noise controls add
three independent teachers; repeated shot budgets, distributions, and precision
settings do not increase the count of independent teachers.

The [figures](../docs/figures) are generated from these records. Their manifest
fingerprints the plotting source, numerical inputs, and SVG/PDF/PNG artifacts.
The [reconstruction animation](../docs/media/reconstruction.gif) uses one
recorded Qiskit case; its [manifest](../docs/media/manifest.json) identifies
every displayed training row, the reconstruction trace, renderer and media hashes.
Write new runs into `runs/benchmark` to keep the recorded benchmark unchanged.
See [reproduction](../docs/reproduction.md).
