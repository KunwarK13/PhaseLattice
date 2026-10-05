"""Evaluate reconstruction and optimization on the fixed 44-model cohort."""

import argparse
import importlib.metadata
import platform
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from phaselattice import PeriodicModel, default_family, reconstruct
from phaselattice.baselines.least_squares import fit as fit_discrete
from phaselattice.baselines.transfer import refine
from phaselattice.baselines.variable_projection import fit as fit_profiled
from phaselattice.evaluation import quantum_roundtrip, reference_derivatives, relative_error, score
from phaselattice.io import read_json, sha256, write_json
from phaselattice.screening import response_range
from phaselattice.verification import verify_trace

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = read_json(ROOT / "benchmarks/protocol.json")


def provenance(output, device, cases):
    sources = sorted((ROOT / "src/phaselattice").rglob("*.py"))
    sources += [Path(__file__), ROOT / "benchmarks/protocol.json"]
    record = dict(
        schema="phaselattice.run/v1",
        device=device,
        cases=[case["id"] for case in cases],
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
        family_sha256=sha256(default_family()),
        cohort_sha256=sha256(ROOT / "evidence/cases.json"),
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "sympy", "mpmath", "fpylll", "qiskit", "torch")
        },
    )
    path = output / "manifest.json"
    if path.exists():
        previous = read_json(path)
        if any(previous[key] != value for key, value in record.items()):
            raise ValueError(
                "Source, configuration, data or environment changed; use a new output directory"
            )
    else:
        record["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        record["gpu"] = torch.cuda.get_device_name() if device.startswith("cuda") else None
        write_json(path, record)
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for source in sources:
                archive.add(source, arcname=str(source.relative_to(ROOT)))
    torch.set_num_threads(2)
    if device.startswith("cuda"):
        value = torch.eye(16, device=device, dtype=torch.float64)
        torch.linalg.solve(value, value)
        torch.cuda.synchronize()


def cached(path, public, function):
    if path.exists():
        result = read_json(path)
    else:
        result = function()
        write_json(path, result)
    if result["public_sha256"] != sha256(public):
        raise ValueError("Result does not belong to these observations")
    return result


def portfolio(results):
    valid = [result for result in results if result["best_validation_mse"] is not None]
    selected = min(valid, key=lambda result: result["best_validation_mse"])
    return dict(
        method="optimizer-portfolio",
        status=selected["status"],
        model=selected["model"],
        best_validation_mse=selected["best_validation_mse"],
        selected_method=selected["method"],
        seconds=sum(result["seconds"] for result in results),
        public_sha256=selected["public_sha256"],
        selection="validation MSE",
        uses_oracle=False,
    )


def evaluate_cases(output, device, cases, stage):
    rows = []
    for specification in cases:
        identifier = specification["id"]
        case = ROOT / "evidence/cases" / identifier
        public = case / "observations.json.gz"
        destination = output / identifier
        if stage in ("all", "recovery"):
            result = cached(
                destination / "reconstruction.json", public, lambda: reconstruct(public).record
            )
            if result["model"] is not None:
                check = verify_trace(public, result)
                if not check["valid"]:
                    raise AssertionError(check)
                expected = read_json(case / "expected_model.json")
                if result["model"] != expected:
                    raise AssertionError(f"Numerical reconstruction changed for {identifier}")
        if stage in ("all", "optimizers"):
            initial = []
            for method in ("multistart-lm", "phase-projection"):
                initial.append(
                    cached(
                        destination / (method + ".json"),
                        public,
                        lambda method=method: fit_discrete(
                            public,
                            default_family(),
                            method,
                            device=device,
                            **PROTOCOL["discrete_optimizer"],
                        ),
                    )
                )
            transferred = cached(
                destination / "optimizer-transfer.json",
                public,
                lambda: refine(public, default_family(), initial, device),
            )
            profiled = cached(
                destination / "variable-projection.json",
                public,
                lambda: fit_profiled(
                    public,
                    default_family(),
                    device=device,
                    range_screen=True,
                    **PROTOCOL["variable_projection"],
                ),
            )
            write_json(destination / "optimizer-portfolio.json", portfolio([transferred, profiled]))
        for method in (
            "reconstruction",
            "multistart-lm",
            "phase-projection",
            "optimizer-transfer",
            "variable-projection",
            "optimizer-portfolio",
        ):
            path = destination / (method + ".json")
            if not path.exists():
                continue
            result = read_json(path)
            row = dict(
                id=identifier,
                cohort=specification["cohort"],
                provider=specification["provider"],
                dimension=specification["dimension"],
                method=method,
                seconds=result["seconds"],
                validation_mse=result["best_validation_mse"],
                **score(case, result),
            )
            rows.append(row)
        write_json(output / "summary.json", dict(schema="phaselattice.summary/v1", rows=rows))
        completed = [row for row in rows if row["id"] == identifier]
        print(
            identifier,
            " ".join(f"{row['method']}={int(row['recovered'])}" for row in completed),
            flush=True,
        )


def derivatives(output, cases):
    rows = []
    for index, specification in enumerate(cases):
        identifier = specification["id"]
        case = ROOT / "evidence/cases" / identifier
        destination = output / identifier
        reference = read_json(case / "reference.json")
        dimension = reference["d"]
        rng = np.random.default_rng(951000 + index)
        n = PROTOCOL["derivative_points"]
        inputs = [
            ("gaussian", rng.normal(size=(n, dimension))),
            ("scaled_gaussian", 3 * rng.normal(size=(n, dimension))),
            ("translated_gaussian", 2 + rng.normal(size=(n, dimension))),
            ("uniform", rng.uniform(-3, 3, size=(n, dimension))),
        ]
        case_rows = []
        for method in ("reconstruction", "variable-projection", "optimizer-portfolio"):
            result = read_json(destination / (method + ".json"))
            if result["model"] is None:
                case_rows.append(
                    dict(id=identifier, method=method, passed=False, status="UNRESOLVED")
                )
                continue
            model = PeriodicModel.from_record(result["model"])
            errors = []
            for distribution, values in inputs:
                vectors = np.random.default_rng(952000 + index).normal(size=values.shape)
                y, gradient, hvp = reference_derivatives(reference, values, vectors)
                errors.append(
                    dict(
                        distribution=distribution,
                        points=n,
                        output_relative_rms_error=relative_error(model.predict(values), y),
                        gradient_relative_rms_error=relative_error(
                            model.gradient(values), gradient
                        ),
                        hvp_relative_rms_error=relative_error(
                            model.hessian_vector_product(values, vectors), hvp
                        ),
                    )
                )
            row = dict(
                id=identifier,
                method=method,
                distributions=errors,
                passed=all(
                    error[key] < PROTOCOL["derivative_relative_error"]
                    for error in errors
                    for key in ("gradient_relative_rms_error", "hvp_relative_rms_error")
                ),
            )
            if method == "reconstruction" and reference["provider"] == "qiskit":
                row["qiskit"] = quantum_roundtrip(model, reference, inputs[2][1][:8])
            case_rows.append(row)
        write_json(destination / "derivatives.json", dict(seed=951000 + index, rows=case_rows))
        rows.extend(case_rows)
        write_json(output / "derivatives.json", dict(rows=rows))
        print(identifier, "derivatives", case_rows[0]["passed"], flush=True)


def timings(output):
    rows = []
    for dimension in (3, 6, 10):
        for index in range(2):
            identifier = f"analytic_d{dimension:02d}_s{index:02d}"
            public = ROOT / "evidence/cases" / identifier / "observations.json.gz"
            results = {}
            for screen in [False, True] if index == 0 else [True, False]:
                response_range.cache_clear()
                name = "screened" if screen else "full"
                result = cached(
                    output / "timing" / identifier / (name + ".json"),
                    public,
                    lambda screen=screen: reconstruct(public, screen=screen).record,
                )
                if not verify_trace(public, result)["valid"]:
                    raise AssertionError("Invalid timing reconstruction")
                results[name] = result
            if results["full"]["model"] != results["screened"]["model"]:
                raise AssertionError("Range filtering changed the recovered equation")
            row = dict(
                id=identifier,
                dimension=dimension,
                full_seconds=results["full"]["seconds"],
                screened_seconds=results["screened"]["seconds"],
                speed_ratio=results["full"]["seconds"] / results["screened"]["seconds"],
                retained_candidates=results["screened"]["screening"]["retained_candidates"],
                identical_model=True,
                order=["full", "screened"] if index == 0 else ["screened", "full"],
            )
            rows.append(row)
            write_json(output / "timing.json", dict(rows=rows))
            print(identifier, f"filtering {row['speed_ratio']:.2f}x", flush=True)


def coarsen(record, precision, shuffle=False):
    record = {**record, "train": dict(record["train"]), "validation": dict(record["validation"])}
    divisor = 1 << (record["B"] - precision)

    def rounded(value):
        return (
            (1 if value >= 0 else -1) * ((abs(value) + divisor // 2) // divisor)
            if divisor > 1
            else value
        )

    for name in ("train", "validation"):
        record[name] = dict(
            X=[[rounded(v) for v in row] for row in record[name]["X"]],
            y=[rounded(v) for v in record[name]["y"]],
        )
        if shuffle:
            np.random.default_rng(121 if name == "train" else 122).shuffle(record[name]["y"])
    record["B"] = precision
    return record


def controls(output, device):
    precision_rows, ablation_rows, noise_rows = [], [], []
    for dimension in (3, 6, 10):
        for index in range(2):
            identifier = f"analytic_d{dimension:02d}_s{index:02d}"
            case = ROOT / "evidence/cases" / identifier
            original = read_json(case / "observations.json.gz")
            for precision in PROTOCOL["precision_sweep"]:
                public = output / "controls/precision" / f"{identifier}_b{precision}.json.gz"
                write_json(public, coarsen(original, precision))
                result = cached(
                    public.with_suffix(".result.json"), public, lambda: reconstruct(public).record
                )
                precision_rows.append(
                    dict(
                        id=identifier,
                        dimension=dimension,
                        precision=precision,
                        seconds=result["seconds"],
                        **score(case, result),
                    )
                )
            for mode in ("single_relation", "shuffled"):
                public = output / "controls/ablation" / f"{identifier}_{mode}.json.gz"
                write_json(public, coarsen(original, 40, shuffle=mode == "shuffled"))
                result = cached(
                    public.with_suffix(".result.json"),
                    public,
                    lambda mode=mode: (
                        reconstruct(
                            public, relations="single" if mode == "single_relation" else "full"
                        ).record
                    ),
                )
                ablation_rows.append(
                    dict(id=identifier, dimension=dimension, control=mode, **score(case, result))
                )
            write_json(output / "precision.json", dict(rows=precision_rows))
            write_json(output / "ablations.json", dict(rows=ablation_rows))
            print(identifier, "precision and ablation controls", flush=True)
    for shots in PROTOCOL["finite_shots"]:
        for index in range(3):
            case = ROOT / "evidence/controls" / f"shots_{shots}_s{index}"
            public = case / "observations.json.gz"
            result = cached(
                output / "controls" / case.name / "reconstruction.json",
                public,
                lambda: reconstruct(public, screen=False).record,
            )
            fitted = cached(
                output / "controls" / case.name / "variable-projection.json",
                public,
                lambda: fit_profiled(
                    public,
                    default_family(),
                    device=device,
                    range_screen=False,
                    **PROTOCOL["variable_projection"],
                ),
            )
            outcome = score(case, fitted)
            noise_rows.append(
                dict(
                    shots=shots,
                    seed_index=index,
                    lattice_status=result["status"],
                    approximate_prediction=outcome["loss_over_variance"] < 1e-3,
                    **outcome,
                )
            )
            write_json(output / "noise.json", dict(independent_teachers=3, rows=noise_rows))
            print(case.name, "noise control", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage",
        choices=("all", "recovery", "optimizers", "derivatives", "timing", "controls"),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = read_json(ROOT / "evidence/cases.json")["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]
    provenance(args.output, args.device, cases)
    if args.stage in ("all", "recovery", "optimizers"):
        evaluate_cases(args.output, args.device, cases, args.stage)
    if args.stage in ("all", "derivatives"):
        derivatives(args.output, cases)
    if args.stage in ("all", "timing"):
        timings(args.output)
    if args.stage in ("all", "controls"):
        controls(args.output, args.device)


if __name__ == "__main__":
    main()
