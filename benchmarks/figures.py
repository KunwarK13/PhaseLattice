"""Render figures and the results page from recorded measurements."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from phaselattice import PeriodicModel
from phaselattice.io import read_json, sha256, write_json
from phaselattice.observations import load_public

ROOT = Path(__file__).resolve().parents[1]
INK = "#233449"
TEAL = "#007F78"
AMBER = "#B26829"
SLATE = "#81909D"
GRID = "#DFE4E6"
PAPER = "#FAFAF7"
FONT = "DejaVu Sans"
GROUPS = (
    [("original", "analytic", d, f"Analytic · {d} weights") for d in (3, 6, 10)]
    + [("original", "qiskit", d, f"Qiskit · {d} weights") for d in (4, 8)]
    + [("extension", None, d, f"Extension · {d} weights") for d in (4, 8, 12)]
)
METHODS = {
    "reconstruction": ("PhaseLattice", TEAL),
    "optimizer-portfolio": ("Combined optimizers", AMBER),
    "variable-projection": ("Variable projection", SLATE),
}


def style():
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.titleweight": "medium",
            "axes.titlepad": 15,
            "svg.fonttype": "none",
            "svg.hashsalt": "phaselattice-v1",
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def heading(fig, title, subtitle):
    fig.text(0.065, 0.945, title, fontsize=19, weight="medium", va="top")
    fig.text(0.065, 0.882, subtitle, fontsize=10, color=SLATE, va="top")


def save(fig, destination, name):
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        metadata = {"Creator": "PhaseLattice"}
        if extension == "svg":
            metadata["Date"] = None
        elif extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(destination / f"{name}.{extension}", dpi=180, metadata=metadata)
    plt.close(fig)


def members(rows, group, method):
    cohort, provider, dimension, _ = group
    return [
        row
        for row in rows
        if row["method"] == method
        and row["cohort"] == cohort
        and row["dimension"] == dimension
        and (provider is None or row["provider"] == provider)
    ]


def overview(run, destination):
    identifier = "analytic_d03_s01"
    record = read_json(run / identifier / "reconstruction.json")
    model = PeriodicModel.from_record(record["model"])
    _, splits = load_public(ROOT / "evidence/cases" / identifier / "observations.json.gz")
    train = splits["train"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.065, right=0.965, top=0.72, bottom=0.24, wspace=0.37)
    heading(
        fig,
        "Periodic model reconstruction",
        "Unknown weights and waveform  /  integer relations  /  differentiable exports",
    )
    left, middle, right = axes
    left.scatter(train.inputs[:384, 0], train.outputs[:384], s=10, color=SLATE, alpha=0.5)
    left.set(xlabel="First input coordinate", ylabel="Response", title="01  Observations")
    left.set_ylim(-1.07, 1.07)
    trace = record["trace"]
    decoded = [
        lift + sign * float(Fraction(phases[branch]))
        for phases, branch, sign, lift in zip(
            trace["phases"], trace["branches"], trace["signs"], trace["lifts"]
        )
    ]
    winding_bound = int(np.ceil(max(abs(value) for value in decoded)))
    for row, (phases, branch, sign, lift) in enumerate(
        zip(trace["phases"], trace["branches"], trace["signs"], trace["lifts"])
    ):
        possibilities = [
            winding + orientation * float(Fraction(phase))
            for phase in phases
            for winding in range(-winding_bound, winding_bound + 1)
            for orientation in (-1, 1)
        ]
        middle.scatter(possibilities, np.full(len(possibilities), row), s=9, color=GRID)
        selected = lift + sign * float(Fraction(phases[branch]))
        middle.scatter(selected, row, s=38, color=TEAL, zorder=3)
    middle.set(
        xlim=(-winding_bound - 0.1, winding_bound + 0.1),
        xlabel="Unwrapped projection",
        ylabel="Selected observation",
        title="02  Phase assignments",
        yticks=range(len(trace["indices"])),
        yticklabels=range(1, len(trace["indices"]) + 1),
    )
    middle.invert_yaxis()
    middle.set_xticks([-winding_bound, 0, winding_bound])
    phase = (train.inputs @ model.theta) % 1
    right.scatter(phase[:384], train.outputs[:384], s=12, color=SLATE, alpha=0.45)
    t = np.linspace(0, 1, 1200)
    waveform = sum(
        k * np.cos(2 * np.pi * j * t) / model.denominator
        for j, k in enumerate(model.coefficients, 1)
    )
    right.plot(t, waveform, color=TEAL, linewidth=2)
    right.set(
        xlim=(0, 1),
        ylim=(-1.07, 1.07),
        xlabel="Recovered phase · modulo one",
        ylabel="Response",
        title="03  Reconstructed model",
    )
    right.set_xticks([0, 0.5, 1])
    for ax in axes:
        ax.tick_params(length=0, pad=7)
    fig.text(
        0.065,
        0.065,
        "Saved observations and reconstruction trace. The center panel shows phase choices "
        "for one candidate waveform; teal marks the decoded choices.",
        fontsize=8.6,
        color=SLATE,
    )
    save(fig, destination, "overview")


def recovery(rows, destination):
    counts = Counter(row["method"] for row in rows if row["recovered"])
    total = sum(row["method"] == "reconstruction" for row in rows)
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    fig.subplots_adjust(left=0.235, right=0.95, top=0.735, bottom=0.14)
    heading(
        fig,
        "Reconstruction success by input dimension",
        f"{counts['reconstruction']}/{total} PhaseLattice  ·  "
        f"{counts['optimizer-portfolio']}/{total} combined optimizers  ·  "
        "40 fractional bits for every method",
    )
    offsets = [-0.23, 0, 0.23]
    for offset, (method, (label, color)) in zip(offsets, METHODS.items()):
        for i, group in enumerate(GROUPS):
            subset = members(rows, group, method)
            n, successes = len(subset), sum(row["recovered"] for row in subset)
            ax.barh(i + offset, successes / n, height=0.17, color=color)
            ax.text(
                successes / n + 0.017,
                i + offset,
                f"{successes}/{n}",
                fontsize=9,
                va="center",
                color=color,
            )
    ax.set(yticks=range(len(GROUPS)), yticklabels=[group[3] for group in GROUPS], xlim=(0, 1.12))
    ax.invert_yaxis()
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Fraction meeting all reconstruction criteria", labelpad=11)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, pad=10)
    for separator in (2.5, 4.5):
        ax.axhline(separator, color=GRID, linewidth=0.8)
    handles = [
        Line2D([0], [0], color=color, lw=5, label=label) for label, color in METHODS.values()
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.23, 0.83), ncol=3, frameon=False
    )
    fig.text(
        0.065,
        0.032,
        "Success: correct normalized waveform, weight error < 10⁻⁴ up to sign, "
        "and independent test MSE/variance < 10⁻⁶. Counts are individual teachers.",
        fontsize=8.8,
        color=SLATE,
    )
    save(fig, destination, "recovery")


def derivatives(rows, destination):
    recovered = [row for row in rows if row["method"] == "reconstruction"]
    distributions = ["gaussian", "scaled_gaussian", "translated_gaussian", "uniform"]
    metrics = [
        ("output_relative_rms_error", "Output", SLATE, "o"),
        ("gradient_relative_rms_error", "Gradient", TEAL, "s"),
        ("hvp_relative_rms_error", "Hessian–vector", AMBER, "D"),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.65, bottom=0.22, wspace=0.65)
    heading(
        fig,
        "Derivative and circuit reconstruction errors",
        "No gradient observations during learning  ·  independent autograd and circuit checks",
    )
    for offset, (metric, label, color, marker) in zip([-0.18, 0, 0.18], metrics):
        maxima = [
            max(
                item[metric]
                for row in recovered
                for item in row["distributions"]
                if item["distribution"] == distribution
            )
            for distribution in distributions
        ]
        left.scatter(maxima, np.arange(4) + offset, color=color, marker=marker, s=40, label=label)
    left.set(
        xscale="log",
        yticks=range(4),
        yticklabels=["Gaussian", "3 × Gaussian", "Translated Gaussian", "Uniform [−3, 3]"],
        xlabel="Maximum relative RMS error",
        title=f"Derivative checks · {len(recovered)} models",
    )
    left.invert_yaxis()
    quantum = [row["qiskit"] for row in recovered if "qiskit" in row]
    for position, (key, color) in enumerate(
        [("max_circuit_difference", TEAL), ("gradient_relative_rms_error", AMBER)]
    ):
        values = [row[key] for row in quantum]
        right.scatter(values, position + np.linspace(-0.12, 0.12, len(values)), color=color, s=30)
    right.set(
        xscale="log",
        yticks=[0, 1],
        yticklabels=["Response\nabsolute error", "Gradient\nrelative RMS"],
        xlabel="Error · one point per teacher",
        title=f"Qiskit round-trip · {len(quantum)} teachers",
        ylim=(1.5, -0.5),
    )
    for ax in (left, right):
        ax.set_xlim(1e-14, 1e-5)
        ax.set_xticks([1e-13, 1e-10, 1e-7])
        ax.axvline(1e-6, ls="--", lw=0.8, color=SLATE)
        ax.tick_params(length=0, pad=8)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.spines[["left", "bottom"]].set_visible(False)
    fig.legend(
        *left.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.16, 0.815),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.065,
        0.08,
        "Left: 512 inputs per distribution; each marker is the maximum across models. "
        "Right: eight translated inputs per circuit.",
        fontsize=8.8,
        color=SLATE,
    )
    fig.text(
        0.065,
        0.035,
        "Dashed line: 10⁻⁶. Circuit gradients use five-point finite differences. "
        "These are structured, classically simulable circuits.",
        fontsize=8.8,
        color=SLATE,
    )
    save(fig, destination, "derivatives")


def precision(precision_rows, noise_rows, destination):
    teachers = sorted({row["id"] for row in precision_rows})
    bits = sorted({row["precision"] for row in precision_rows})
    lookup = {(row["id"], row["precision"]): row["recovered"] for row in precision_rows}
    matrix = np.array([[lookup[(teacher, bit)] for bit in bits] for teacher in teachers])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.8))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.735, bottom=0.24, wspace=0.52)
    heading(
        fig,
        "Precision and shot-noise sensitivity",
        "The same six teachers across bit depths  /  three additional teachers at two shot budgets",
    )
    left.imshow(matrix, cmap=ListedColormap(["#E9EDEC", TEAL]), aspect="auto", vmin=0, vmax=1)
    for i in range(len(teachers)):
        for j in range(len(bits)):
            left.text(
                j,
                i,
                "●" if matrix[i, j] else "×",
                ha="center",
                va="center",
                color=PAPER if matrix[i, j] else SLATE,
                fontsize=11,
            )
    left.set(
        xticks=range(len(bits)),
        xticklabels=bits,
        yticks=range(len(teachers)),
        yticklabels=[
            f"{int(name.split('_')[1][1:])} weights · {int(name[-2:]) + 1}" for name in teachers
        ],
        xlabel="Fractional bits",
        title="Lattice reconstruction",
    )
    left.tick_params(length=0, pad=8)
    left.spines[:].set_visible(False)
    shots = sorted({row["shots"] for row in noise_rows})
    for seed in sorted({row["seed_index"] for row in noise_rows}):
        paired = sorted(
            [row for row in noise_rows if row["seed_index"] == seed], key=lambda row: row["shots"]
        )
        right.plot(
            range(len(paired)),
            [row["loss_over_variance"] for row in paired],
            "o-",
            color=AMBER,
            alpha=0.75,
            linewidth=1.2,
            markersize=5,
        )
    right.axhline(1e-3, color=SLATE, ls="--", lw=0.8)
    right.set(
        xticks=range(len(shots)),
        xticklabels=[f"{shot:,}" for shot in shots],
        xlim=(-0.3, len(shots) - 0.7),
        yscale="log",
        ylim=(1e-8, 3e-3),
        ylabel="Test MSE / target variance",
        xlabel="Shots per observation",
        title="Variable projection under shot noise",
    )
    right.grid(axis="y", color=GRID, linewidth=0.6)
    right.tick_params(length=0, pad=8)
    unresolved = sum(row["lattice_status"] == "UNRESOLVED" for row in noise_rows)
    fig.text(
        0.065,
        0.10,
        "Left: teal indicates all reconstruction criteria passed. "
        "Right: paired measurements on three teachers; dashed line is the approximate-prediction threshold.",
        fontsize=8.7,
        color=SLATE,
    )
    fig.text(
        0.065,
        0.055,
        f"PhaseLattice returned unresolved on {unresolved}/{len(noise_rows)} noisy datasets. "
        "The high-precision benchmark does not establish hardware-noise robustness.",
        fontsize=8.7,
        color=SLATE,
    )
    save(fig, destination, "precision")


def filtering(rows, destination):
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.subplots_adjust(left=0.2, right=0.86, top=0.735, bottom=0.22)
    median = np.median([row["speed_ratio"] for row in rows])
    heading(
        fig,
        "Inference time with exact range filtering",
        f"Median paired speedup: {median:.2f}×  ·  identical recovered models in every pair",
    )
    for i, row in enumerate(rows):
        a, b = row["screened_seconds"], row["full_seconds"]
        ax.plot([a, b], [i, i], color=GRID, lw=3)
        ax.scatter(a, i, color=TEAL, s=45, zorder=3)
        ax.scatter(b, i, color=SLATE, s=45, zorder=3)
        ax.text(
            1.03,
            i,
            f"{row['speed_ratio']:.2f}×",
            va="center",
            transform=ax.get_yaxis_transform(),
            fontsize=10,
        )
    ax.set(
        yticks=range(len(rows)),
        yticklabels=[f"{row['dimension']} weights · {int(row['id'][-2:]) + 1}" for row in rows],
        xscale="log",
        xlabel="Inference time · seconds",
        ylim=(len(rows) - 0.5, -0.5),
    )
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(length=0, pad=9)
    ax.spines[["left", "bottom"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=label)
        for label, color in [("Range filtered", TEAL), ("Full family", SLATE)]
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.64, 0.83), ncol=2, frameon=False
    )
    fig.text(
        0.065,
        0.06,
        "Six paired measurements on CPU, alternating order and clearing range caches. "
        "Compilation, imports and evaluation are excluded.",
        fontsize=8.6,
        color=SLATE,
    )
    save(fig, destination, "filtering")


def results_page(
    run, rows, derivative_rows, timing_rows, precision_rows, ablation_rows, noise_rows, path
):
    counts = Counter(row["method"] for row in rows if row["recovered"])
    models = [row for row in rows if row["method"] == "reconstruction"]
    derivative_models = [row for row in derivative_rows if row["method"] == "reconstruction"]
    errors = [
        item[key]
        for row in derivative_models
        for item in row["distributions"]
        for key in ("gradient_relative_rms_error", "hvp_relative_rms_error")
    ]
    quantum = [row["qiskit"] for row in derivative_models if "qiskit" in row]
    table = [
        "| Cohort | PhaseLattice | Variable projection | Combined optimizers |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        cells = []
        for method in ("reconstruction", "variable-projection", "optimizer-portfolio"):
            subset = members(rows, group, method)
            cells.append(f"{sum(row['recovered'] for row in subset)}/{len(subset)}")
        table.append(f"| {group[3]} | " + " | ".join(cells) + " |")
    table.append(
        f"| **Total** | **{counts['reconstruction']}/{len(models)}** | "
        f"**{counts['variable-projection']}/{len(models)}** | "
        f"**{counts['optimizer-portfolio']}/{len(models)}** |"
    )
    precision_table = ["| Fractional bits | Recovered |", "|---:|---:|"]
    for bit in sorted({row["precision"] for row in precision_rows}):
        subset = [row for row in precision_rows if row["precision"] == bit]
        precision_table.append(
            f"| {bit} | {sum(row['recovered'] for row in subset)}/{len(subset)} |"
        )
    noise_table = [
        "| Shots | Lattice recovered | Variable projection: approximate prediction | "
        "Variable projection: strict reconstruction | Maximum test MSE/variance |",
        "|---:|---:|---:|---:|---:|",
    ]
    for shot in sorted({row["shots"] for row in noise_rows}):
        subset = [row for row in noise_rows if row["shots"] == shot]
        n = len(subset)
        noise_table.append(
            f"| {shot:,} | {sum(row['lattice_status'] == 'RECOVERED' for row in subset)}/{n} | "
            f"{sum(row['approximate_prediction'] for row in subset)}/{n} | "
            f"{sum(row['recovered'] for row in subset)}/{n} | "
            f"{max(row['loss_over_variance'] for row in subset):.2e} |"
        )
    ablation_counts = {
        control: (
            sum(row["recovered"] for row in ablation_rows if row["control"] == control),
            sum(row["control"] == control for row in ablation_rows),
        )
        for control in ("single_relation", "shuffled")
    }
    median = np.median([row["speed_ratio"] for row in timing_rows])
    manifest = read_json(run / "manifest.json")
    text = f"""# Measured results

This page and its figures are generated from the saved measurements by
`python benchmarks/figures.py`. The [protocol](protocol.md) defines the cohort,
algorithm budgets, thresholds, and timing conventions. The
[evidence index](../evidence/README.md) links to individual records.

## Recovery

{chr(10).join(table)}

![Reconstruction counts](figures/recovery.svg)

Success requires the correct normalized waveform, weight error below `1e-4`
up to sign, and test MSE/variance below `1e-6`. Every method receives the same
1,024 training and 128 validation observations at 40 fractional bits. Test
scoring uses a separate 2,048 observations. Variable projection is also a
member of the combined optimizer portfolio, so these columns are not independent.

PhaseLattice recovered more models than the optimizer portfolio in this benchmark.
The comparison is limited to this model family, observation precision, and
optimization budgets. Successful optimizers can estimate the weights more precisely
than the lattice decoder. The difference in reconstruction rates is largest at
higher input dimensions.

## Derivatives and Qiskit

All {len(derivative_models)} recovered equations are evaluated on four distributions,
with 512 points per distribution. {sum(row["passed"] for row in derivative_models)}/{len(derivative_models)}
pass the gradient and Hessian-vector criterion on every distribution.
The largest relative RMS error across these derivative checks is **{max(errors):.2e}**.

The {len(quantum)} Qiskit teachers use at most three qubits. The largest absolute
response difference between original and reconstructed circuits is
**{max(row["max_circuit_difference"] for row in quantum):.2e}**; the largest gradient
relative RMS error against circuit finite differences is
**{max(row["gradient_relative_rms_error"] for row in quantum):.2e}**.
The circuit reconstruction reproduces the measured response; the underlying
gate sequence is not identifiable from these observations.

![Derivative and circuit checks](figures/derivatives.svg)

The circuits are structured and classically simulable. These checks test response
and gradient agreement within the specified circuit family. They do not
demonstrate quantum advantage or reconstruction of arbitrary quantum models.

## Exact filtering

On six paired analytic cases, removing range-incompatible waveforms gives a
**{median:.2f}× median speedup** over the same decoder with filtering disabled.
Every pair returns the same rational model and a valid reconstruction trace.
Each pair was measured once on the recorded workstation, with alternating
execution order and range caches cleared before each run. These timings may vary
with hardware and process load.

![Paired filtering times](figures/filtering.svg)

## Precision and negative controls

{chr(10).join(precision_table)}

Each row reuses the same six teachers; this is a paired precision sweep.
The single-relation ablation recovers {ablation_counts["single_relation"][0]}/{ablation_counts["single_relation"][1]}
teachers. Independently shuffled labels recover
{ablation_counts["shuffled"][0]}/{ablation_counts["shuffled"][1]}.

{chr(10).join(noise_table)}

The shot-noise experiment has three independent teachers, each measured at two
shot budgets. Approximate prediction means test MSE/variance below `1e-3`,
whereas strict reconstruction retains all three original criteria.

![Precision and noise controls](figures/precision.svg)

## Recorded environment

The recorded run used Python {manifest["python"]}, NumPy {manifest["versions"]["numpy"]},
PyTorch {manifest["versions"]["torch"]}, and Qiskit {manifest["versions"]["qiskit"]}.
Optimization ran on {manifest["gpu"] or "CPU"}; lattice reduction ran on CPU.
The [manifest](../evidence/run/manifest.json) records source and data hashes,
library versions and platform information. The [audit](../evidence/audit.json)
rechecks observations, reconstruction traces, selection rules, and scores.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "evidence/run")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--results", type=Path, default=ROOT / "docs/results.md")
    parser.add_argument("--overview-only", action="store_true")
    args = parser.parse_args()
    style()
    overview(args.run, args.output)
    if args.overview_only:
        return
    names = ["summary", "derivatives", "timing", "precision", "ablations", "noise"]
    records = {name: read_json(args.run / f"{name}.json")["rows"] for name in names}
    cases = read_json(ROOT / "evidence/cases.json")["cases"]
    if len(records["summary"]) != 6 * len(cases):
        raise ValueError("A complete six-method run is required to render the benchmark figures")
    if len(records["derivatives"]) != 3 * len(cases):
        raise ValueError("Derivative evaluation is incomplete")
    recovery(records["summary"], args.output)
    derivatives(records["derivatives"], args.output)
    precision(records["precision"], records["noise"], args.output)
    filtering(records["timing"], args.output)
    results_page(args.run, *(records[name] for name in names), args.results)
    write_json(
        args.output / "manifest.json",
        {
            "schema": "phaselattice.figures/v1",
            "generator_sha256": sha256(Path(__file__)),
            "inputs": {name + ".json": sha256(args.run / f"{name}.json") for name in names},
            "overview": {
                "case": "analytic_d03_s01",
                "reconstruction_sha256": sha256(args.run / "analytic_d03_s01/reconstruction.json"),
                "public_sha256": sha256(
                    ROOT / "evidence/cases/analytic_d03_s01/observations.json.gz"
                ),
            },
            "artifacts": {
                path.name: sha256(path)
                for path in sorted(args.output.iterdir())
                if path.suffix in (".svg", ".pdf", ".png")
            },
        },
    )
    print(f"Wrote five figures in SVG, PDF and PNG, and {args.results.name}")


if __name__ == "__main__":
    main()
