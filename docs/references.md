# Research context

The numerical core implements **selected phase-list decoding for finite
families of periodic single-index models** from the associated research
manuscript, which is not included in this repository. The code uses certified
bands with one or two folded inverse phases and an LLL decoder. It does not
include the exploratory higher-branch certificates or resonance-discovery routines.

The learning problem is to recover both a weight vector and an unknown waveform
from a known finite family. Inverting one output yields a list of phases. The
decoder resolves these choices jointly using shared integer relations.

The repository implements the saved-observation interface, exact range
filtering, independent trace replay, standalone derivative exports, Qiskit
response reconstruction, optimization controls, and reproducible evaluation.
The experiments are limited to the specified model family and observation settings.

## Relevant prior work

- Min Jae Song, Ilias Zadik, and Joan Bruna. *On the Cryptographic Hardness of
  Learning Single Periodic Neurons* (2021).
  [Paper](https://arxiv.org/abs/2106.10744). Establishes prior lattice-based
  learning for periodic neurons and the importance of the noise/access model.
- Ilias Diakonikolas and Daniel Kane. *Non-Gaussian Component Analysis via
  Lattice Basis Reduction* (2022).
  [Paper](https://proceedings.mlr.press/v178/diakonikolas22d.html). Recovers hidden
  non-Gaussian structure using lattice reduction; a precedent for resolving
  discrete structure through the full kernel of the observation matrix.
- Dianne P. O'Leary and Bert W. Rust. *Variable Projection for Nonlinear Least
  Squares Problems*.
  [Paper](https://www.cs.umd.edu/users/oleary/software/varpro.pdf). Describes the
  optimization method used in the variable-projection baseline.
- Franz J. Schreiber, Jens Eisert, and Johannes Jakob Meyer. *Classical Surrogates
  for Quantum Learning Models* (2022).
  [Paper](https://arxiv.org/abs/2206.11740). Studies classical surrogates for
  quantum learning models.
- Zhenxiao Fu and Fan Chen. *Quantum Neural Network Extraction Attack via Split
  Co-Teaching* (2024; revised 2025).
  [Paper](https://arxiv.org/abs/2409.02207). Demonstrates prior QNN extraction
  research in a different noise and learning setting.

Quantum model extraction, classical surrogates, and gradient export have prior
art. This project's evaluation concerns selected phase-list decoding with an
unknown waveform in a finite family.
