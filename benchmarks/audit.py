"""Recompute evidence checks without fitting models or rerunning lattice reduction."""

import argparse
import ast
import hashlib
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from phaselattice import default_family
from phaselattice.evaluation import score
from phaselattice.io import content_digest, read_json, sha256, write_json
from phaselattice.observations import load_public
from phaselattice.verification import verify_trace

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "reconstruction",
    "multistart-lm",
    "phase-projection",
    "optimizer-transfer",
    "variable-projection",
    "optimizer-portfolio",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_score(actual, saved, context):
    for key, value in actual.items():
        if isinstance(value, float):
            require(np.isclose(value, saved[key], rtol=1e-10, atol=1e-30), f"{context}: {key}")
        else:
            require(value == saved[key], f"{context}: {key}")


def executable_ast(source):
    """Parse Python source without comments, docstrings, or location metadata."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def audit(run, figure_directory=None, media_directory=None):
    manifest = read_json(run / "manifest.json")
    documentation_path = ROOT / "evidence/source_documentation.json"
    documentation = {}
    if documentation_path.exists():
        revisions = read_json(documentation_path)
        if revisions["run_manifest_sha256"] == sha256(run / "manifest.json"):
            require(
                revisions["schema"] == "phaselattice.source-documentation/v1",
                "Unknown source documentation schema",
            )
            documentation = revisions["files"]
            require(
                set(documentation).issubset(manifest["source_sha256"]),
                "Documentation revision names an unrecorded source file",
            )
    with tarfile.open(run / "source.tar.gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        require(names == set(manifest["source_sha256"]), "Source archive member mismatch")
        for member in archive.getmembers():
            if member.isfile():
                original = archive.extractfile(member).read()
                digest = hashlib.sha256(original).hexdigest()
                require(digest == manifest["source_sha256"][member.name], "Source archive mismatch")
                current = (ROOT / member.name).read_bytes()
                current_digest = hashlib.sha256(current).hexdigest()
                if member.name in documentation:
                    revision = documentation[member.name]
                    require(Path(member.name).suffix == ".py", "Non-Python documentation change")
                    require(revision["benchmark_sha256"] == digest, "Documentation base mismatch")
                    require(
                        revision["current_sha256"] == current_digest,
                        f"Documented source hash mismatch: {member.name}",
                    )
                    require(
                        executable_ast(original) == executable_ast(current),
                        f"Executable source changed: {member.name}",
                    )
                else:
                    require(current_digest == digest, f"Source changed: {member.name}")
    require(sha256(default_family()) == manifest["family_sha256"], "Family changed")
    require(sha256(ROOT / "evidence/cases.json") == manifest["cohort_sha256"], "Cohort changed")
    cases = read_json(ROOT / "evidence/cases.json")["cases"]
    require([case["id"] for case in cases] == manifest["cases"], "Incomplete cohort")
    migration = {row["id"]: row for row in read_json(ROOT / "evidence/migration.json")["records"]}
    summary = read_json(run / "summary.json")["rows"]
    lookup = {(row["id"], row["method"]): row for row in summary}
    require(
        len(lookup) == len(summary) == len(cases) * len(METHODS), "Incomplete or duplicate summary"
    )
    successes = Counter()
    traces = []
    for specification in cases:
        identifier = specification["id"]
        case = ROOT / "evidence/cases" / identifier
        public = case / "observations.json.gz"
        for file, key in [
            ("observations.json.gz", "public_sha256"),
            ("reference.json", "reference_sha256"),
            ("test.npz", "test_sha256"),
        ]:
            require(sha256(case / file) == specification[key], f"Data changed: {identifier}/{file}")
        record, splits = load_public(public)
        numerical = content_digest({key: value for key, value in record.items() if key != "schema"})
        require(
            numerical
            == specification["numerical_sha256"]
            == migration[identifier]["numerical_sha256"],
            f"Numerical migration mismatch: {identifier}",
        )
        require(
            sha256(public) == migration[identifier]["migrated_observations_sha256"],
            f"Migration file mismatch: {identifier}",
        )
        require(
            len(splits["train"].outputs) == specification["training_rows"], "Training row count"
        )
        require(
            len(splits["validation"].outputs) == specification["validation_rows"],
            "Validation row count",
        )
        require(splits["train"].inputs.shape[1] == specification["dimension"], "Input dimension")
        results = {}
        for method in METHODS:
            result = read_json(run / identifier / f"{method}.json")
            require(
                result["public_sha256"] == specification["public_sha256"], "Observation mismatch"
            )
            result_score = score(case, result)
            equal_score(result_score, lookup[(identifier, method)], f"{identifier}/{method}")
            successes[method] += result_score["recovered"]
            results[method] = result
        inferred = results["reconstruction"]
        require(
            inferred["model"] == read_json(case / "expected_model.json"), "Regression model changed"
        )
        check = verify_trace(public, inferred)
        require(check["valid"], f"Invalid reconstruction trace: {identifier}")
        traces.append(dict(id=identifier, **check))
        candidates = [results[method] for method in ("optimizer-transfer", "variable-projection")]
        selected = min(candidates, key=lambda result: result["best_validation_mse"])
        portfolio = results["optimizer-portfolio"]
        require(portfolio["model"] == selected["model"], "Portfolio selection used wrong candidate")
        require(portfolio["selected_method"] == selected["method"], "Portfolio method mismatch")
        require(
            portfolio["best_validation_mse"] == selected["best_validation_mse"],
            "Portfolio validation loss",
        )
        require(
            np.isclose(portfolio["seconds"], sum(result["seconds"] for result in candidates)),
            "Portfolio does not count all constituent work",
        )

    timing = read_json(run / "timing.json")["rows"]
    require(len(timing) == 6, "Incomplete paired timings")
    for row in timing:
        path = run / "timing" / row["id"]
        full, screened = (read_json(path / f"{mode}.json") for mode in ("full", "screened"))
        require(full["model"] == screened["model"], "Filtering changed model")
        require(
            np.isclose(full["seconds"] / screened["seconds"], row["speed_ratio"]), "Timing ratio"
        )
        public = ROOT / "evidence/cases" / row["id"] / "observations.json.gz"
        for result in (full, screened):
            require(verify_trace(public, result)["valid"], "Invalid paired timing trace")

    precision = read_json(run / "precision.json")["rows"]
    ablations = read_json(run / "ablations.json")["rows"]
    require(
        len(precision) == 36 and len(ablations) == 12, "Incomplete precision or ablation controls"
    )
    for name, rows in (("precision", precision), ("ablation", ablations)):
        for row in rows:
            suffix = f"b{row['precision']}" if name == "precision" else row["control"]
            public = run / "controls" / name / f"{row['id']}_{suffix}.json.gz"
            result = read_json(public.with_suffix(".result.json"))
            require(sha256(public) == result["public_sha256"], "Control observation mismatch")
            equal_score(score(ROOT / "evidence/cases" / row["id"], result), row, public.name)
            if result["status"] == "RECOVERED":
                require(verify_trace(public, result)["valid"], "Invalid control trace")

    noise = read_json(run / "noise.json")["rows"]
    require(len(noise) == 6, "Incomplete noise controls")
    for row in noise:
        identifier = f"shots_{row['shots']}_s{row['seed_index']}"
        case = ROOT / "evidence/controls" / identifier
        public = case / "observations.json.gz"
        inferred = read_json(run / "controls" / identifier / "reconstruction.json")
        fitted = read_json(run / "controls" / identifier / "variable-projection.json")
        require(
            inferred["public_sha256"] == fitted["public_sha256"] == sha256(public),
            "Noise data mismatch",
        )
        require(inferred["status"] == row["lattice_status"], "Noise status mismatch")
        outcome = score(case, fitted)
        equal_score(outcome, row, identifier)
        require(
            (outcome["loss_over_variance"] < 1e-3) == row["approximate_prediction"],
            "Noise prediction criterion",
        )

    derivatives = read_json(run / "derivatives.json")["rows"]
    require(len(derivatives) == 3 * len(cases), "Incomplete derivative records")
    require(
        len({(row["id"], row["method"]) for row in derivatives}) == len(derivatives),
        "Duplicate derivative record",
    )
    for specification in cases:
        identifier = specification["id"]
        rows = read_json(run / identifier / "derivatives.json")["rows"]
        require(
            rows == [row for row in derivatives if row["id"] == identifier],
            "Derivative summary mismatch",
        )
        for row in rows:
            if "distributions" in row:
                passed = all(
                    value[key] < 1e-6
                    for value in row["distributions"]
                    for key in ("gradient_relative_rms_error", "hvp_relative_rms_error")
                )
                require(passed == row["passed"], "Derivative threshold mismatch")

    figure_count = 0
    if figure_directory is None and run.resolve() == ROOT / "evidence/run":
        figure_directory = ROOT / "docs/figures"
    if figure_directory is not None:
        figure_manifest = figure_directory / "manifest.json"
        figures = read_json(figure_manifest)
        require(
            figures["generator_sha256"] == sha256(ROOT / "benchmarks/figures.py"),
            "Figure code changed",
        )
        for name, digest in figures["inputs"].items():
            require(sha256(run / name) == digest, f"Figure data mismatch: {name}")
        overview = figures["overview"]
        require(
            sha256(run / overview["case"] / "reconstruction.json")
            == overview["reconstruction_sha256"],
            "Overview reconstruction mismatch",
        )
        require(
            sha256(ROOT / "evidence/cases" / overview["case"] / "observations.json.gz")
            == overview["public_sha256"],
            "Overview observations mismatch",
        )
        for name, digest in figures["artifacts"].items():
            require(sha256(figure_manifest.parent / name) == digest, f"Figure changed: {name}")
            figure_count += 1

    media_count = 0
    if media_directory is None and run.resolve() == ROOT / "evidence/run":
        media_directory = ROOT / "docs/media"
    if media_directory is not None:
        media = read_json(media_directory / "manifest.json")
        require(
            media["generator_sha256"] == sha256(ROOT / "benchmarks/animate.py"),
            "Animation renderer changed",
        )
        require(
            media["cohort_sha256"] == sha256(ROOT / "evidence/cases.json"),
            "Animation cohort mismatch",
        )
        case = media["case"]
        public = ROOT / "evidence/cases" / case / "observations.json.gz"
        result_path = run / case / "reconstruction.json"
        require(media["public_sha256"] == sha256(public), "Animation observations mismatch")
        require(media["reconstruction_sha256"] == sha256(result_path), "Animation trace mismatch")
        result = read_json(result_path)
        require(
            media["selected_training_rows"] == result["trace"]["indices"],
            "Animation selected different rows",
        )
        shown = media["shown_training_rows"]
        require(len(set(shown)) == len(shown), "Duplicate animation observations")
        require(
            all(0 <= row < result["raw_train_rows"] for row in shown),
            "Animation observation outside training data",
        )
        require(set(result["trace"]["indices"]).issubset(shown), "Animation omitted selected rows")
        expected = {f"reconstruction.{suffix}" for suffix in ("gif", "mp4", "png", "svg")}
        require(set(media["artifacts"]) == expected, "Incomplete animation artifacts")
        for name, digest in media["artifacts"].items():
            require(sha256(media_directory / name) == digest, f"Animation changed: {name}")
            media_count += 1

    return dict(
        schema="phaselattice.audit/v1",
        passed=True,
        run_manifest_sha256=sha256(run / "manifest.json"),
        source_files=len(manifest["source_sha256"]),
        source_documentation_files_checked=len(documentation),
        source_documentation_sha256=sha256(documentation_path) if documentation else None,
        independent_models=len(cases),
        method_scores_recomputed=len(summary),
        reconstruction_counts=dict(successes),
        paired_timings=len(timing),
        precision_measurements=len(precision),
        ablation_measurements=len(ablations),
        noisy_measurements=len(noise),
        derivative_records_checked=len(derivatives),
        figure_artifacts_checked=figure_count,
        animation_artifacts_checked=media_count,
        standalone_template_sha256=sha256(ROOT / "src/phaselattice/templates/model.py.tmpl"),
        checks=[
            "Archived source matches the run manifest; current source differs only in recorded comments and docstrings",
            "Public, reference and test files agree with cohort hashes",
            "Canonical numerical observations agree with migration records",
            "Every method score is recomputed on held-out data",
            "Optimizer selection uses validation loss and counts constituent work",
            "Recovered models match the frozen regression records",
            "All recovered traces pass independent rational replay",
            "Precision, noise and ablation scores agree with their result files",
            "Derivative records and thresholds are internally consistent",
        ],
        scope="Integrity and score replay; derivative measurements require the derivatives stage to rerun",
        traces=traces,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "evidence/run")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/audit.json")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    record = audit(args.run, args.figures, args.media)
    write_json(args.output, record)
    print(
        f"Audit passed: {record['independent_models']} models, "
        f"{record['method_scores_recomputed']} recomputed method scores, "
        f"{record['figure_artifacts_checked']} figure artifacts, "
        f"{record['animation_artifacts_checked']} animation artifacts"
    )


if __name__ == "__main__":
    main()
