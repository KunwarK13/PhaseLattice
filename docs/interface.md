# API and data format

## Reconstruction

```python
from phaselattice import reconstruct

result = reconstruct("observations.json.gz")
result.save("reconstruction.json")
check = result.verify("observations.json.gz")
```

`reconstruct` accepts a path to public observations and an optional compiled
waveform family. Range filtering is enabled by default. `screen=False` runs the
full family search. `relations='single'` selects the `d+1`-row ablation.

`result.recovered` distinguishes accepted reconstruction from `UNRESOLVED`.
`result.model` is a `PeriodicModel` for accepted results and `None` otherwise.
`result.record` contains the full trace, attempt diagnostics, data digests, and
timing information. `Reconstruction.load(path)` reads a saved result.

## PeriodicModel

| Operation | Result |
|---|---|
| `predict(X)` | Response for each input |
| `gradient(X)` | Gradient with respect to each input |
| `hessian_vector_product(X, v)` | Hessian applied to `v`, without materializing a matrix |
| `as_torch()` | Float64 module with fixed buffers and input autograd |
| `as_qiskit()` | Circuit, measured observable, and ordered feature parameters |
| `export(path)` | Standalone Python implementation |

The final dimension of `X` must equal the number of hidden weights. A single
input vector and batches are supported. A Hessian-vector product accepts a
single direction or one direction per input. NumPy operations use float64.
The PyTorch module expects matching dtype and device; use `.to(...)` explicitly
when integrating it into another model.

```python
import torch

response = result.model.as_torch()
x = torch.zeros((16, len(result.model.weights)), dtype=torch.float64, requires_grad=True)
response(x).sum().backward()
input_gradients = x.grad
```

Bind Qiskit inputs by parameter identity. This preserves feature order for
dimensions above ten, where lexical sorting of parameter names is insufficient.

```python
circuit, observable, features = result.model.as_qiskit()
bound = circuit.assign_parameters(dict(zip(features, input_vector)))
```

The standalone export supports the same numerical and integration operations.
It requires NumPy; PyTorch and Qiskit are imported only when their integration
functions are called.

## Public observations

The reader accepts JSON and deterministic gzip-compressed JSON. Its exact set
of top-level keys is:

```json
{
  "schema": "phaselattice.observations/v1",
  "B": 40,
  "D": 3,
  "Q": 4,
  "norm_bound": 2.0,
  "train": {"X": [[1099511627776]], "y": [1099511627776]},
  "validation": {"X": [[0]], "y": [1099511627776]}
}
```

This is a format illustration, not enough data for reconstruction. Each input
and output is stored as an integer numerator with implicit denominator `2**B`.
`D` is the largest allowed harmonic degree, `Q` is the coefficient denominator,
and `norm_bound` bounds the hidden weight norm. The modeled class has norm at
least one. The selected waveform belongs to the supplied compiled family.

The reader rejects extra metadata, ragged arrays, inconsistent dimensions,
noninteger numerators, and any value that loses information in float64. This
last check gives the optimization controls the same observation bits as the
rational decoder. Supported precision is 1–44 fractional bits.

Reference weights, generating seeds, and test labels are stored separately and
are accepted only by evaluation utilities.

## CLI exit status

| Code | Meaning |
|---:|---|
| 0 | Command completed successfully |
| 1 | Trace verification failed |
| 2 | Invalid input, missing dependency, or I/O error |
| 3 | Recovery completed without an accepted model |

A recovery attempt writes its result even when unresolved. A model export
requires an accepted reconstruction.
