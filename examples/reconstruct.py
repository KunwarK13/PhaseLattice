"""Reconstruct a frozen Qiskit teacher and export its inferred response."""

from pathlib import Path

from phaselattice import reconstruct
from phaselattice.evaluation import score
from phaselattice.io import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    case = root / "evidence/cases/quantum_d08_s00"
    observations = case / "observations.json.gz"
    destination = root / "runs/example"
    result = reconstruct(observations)
    if not result.recovered:
        raise RuntimeError("The example did not reconstruct its response")
    verification = result.verify(observations)
    if not verification["valid"]:
        raise RuntimeError(verification)
    result.save(destination / "reconstruction.json")
    result.model.export(destination / "model.py")
    evaluation = score(case, result.record)
    write_json(destination / "evaluation.json", evaluation)
    print(f"Recovered {len(result.model.weights)} weights and waveform {result.model.coefficients}")
    print(f"Independent test MSE / variance: {evaluation['loss_over_variance']:.2e}")
    print("Trace verified; standalone model saved to runs/example/model.py")


if __name__ == "__main__":
    main()
