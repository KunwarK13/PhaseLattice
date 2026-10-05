"""A public-file inference process followed by independent evaluation."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from .api import Reconstruction
from .evaluation import quantum_roundtrip, reference_derivatives, relative_error, score
from .family import default_family
from .io import read_json, write_json
from .synthetic import generate


def run_demo(output: str | Path, provider: str = "qiskit") -> dict:
    output = Path(output).resolve()
    if any(output.glob("*.json")):
        raise ValueError("Choose an empty output directory to preserve previous demo results")
    generate(output, default_family(), 4, 720403, provider=provider, n_train=512, n_val=128)
    public = output / "observations.json"
    target = output / "reconstruction.json"
    subprocess.run(
        [sys.executable, "-m", "phaselattice", "recover", str(public), "--output", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = Reconstruction.load(target)
    check = result.verify(public)
    if not check["valid"] or result.model is None:
        raise RuntimeError(f"Demo reconstruction failed: {check}")
    model = result.model
    model.export(output / "model.py")
    reference = read_json(output / "reference.json")
    metrics = score(output, result.record)
    rng = np.random.default_rng(720499)
    inputs = 2 + rng.normal(size=(256, 4))
    vectors = rng.normal(size=inputs.shape)
    values, gradients, hvp = reference_derivatives(reference, inputs, vectors)
    metrics["shifted_inputs"] = dict(
        distribution="N(2*ones,I)",
        points=len(inputs),
        output_relative_rms_error=relative_error(model.predict(inputs), values),
        gradient_relative_rms_error=relative_error(model.gradient(inputs), gradients),
        hvp_relative_rms_error=relative_error(model.hessian_vector_product(inputs, vectors), hvp),
    )
    if provider == "qiskit":
        from qiskit import qpy

        metrics["qiskit"] = quantum_roundtrip(model, reference, inputs[:8])
        circuit, observable, features = model.as_qiskit()
        with (output / "response.qpy").open("wb") as stream:
            qpy.dump(circuit, stream)
        write_json(
            output / "response.json",
            dict(
                input_parameters=[str(p) for p in features],
                observable=[
                    dict(pauli=p, real=float(c.real), imag=float(c.imag))
                    for p, c in observable.to_list()
                ],
            ),
        )
    metrics["trace_verification"] = check
    metrics["reconstruction_seconds"] = result.seconds
    write_json(output / "evaluation.json", metrics)
    return metrics
