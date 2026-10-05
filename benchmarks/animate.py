"""Animate a verified reconstruction using the saved observations and phase trace."""

import argparse
import importlib.metadata
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgb

from phaselattice import PeriodicModel
from phaselattice.io import read_json, sha256, write_json
from phaselattice.observations import load_public
from phaselattice.verification import verify_trace

ROOT = Path(__file__).resolve().parents[1]
PAPER = "#F7F9FF"
PANEL = "#FFFFFF"
INK = "#192840"
TEAL = "#009E87"
INDIGO = "#6254F2"
CORAL = "#F16B42"
CORAL_TEXT = "#B94520"
TEAL_TEXT = "#00766C"
SLATE = "#68768F"
GRID = "#E4E9F3"
SECONDS = 8
VIDEO_FPS = 24
GIF_FPS = 20
POSTER_TIME = 6
CHAPTERS = {
    "observations": 0,
    "phase_choices": 0.8,
    "joint_resolution": 1.2,
    "coordinate_change": 2.4,
    "waveform": 4.6,
    "result": 5.3,
}


def ease(time, start, end):
    value = np.clip((time - start) / (end - start), 0, 1)
    return float(value * value * (3 - 2 * value))


def mix(first, second, amount):
    return (1 - amount) * np.array(to_rgb(first)) + amount * np.array(to_rgb(second))


@dataclass(frozen=True)
class Story:
    case: str
    public_path: Path
    result_path: Path
    record: dict
    model: PeriodicModel
    inputs: np.ndarray
    outputs: np.ndarray
    shown_rows: np.ndarray
    selected: np.ndarray
    projections: np.ndarray
    choices: np.ndarray
    winding_bound: int
    train_count: int
    validation_count: int
    precision: int
    provider: str


def load_story(case, run):
    public_path = ROOT / "evidence/cases" / case / "observations.json.gz"
    result_path = run / case / "reconstruction.json"
    record = read_json(result_path)
    check = verify_trace(public_path, record)
    if not check["valid"]:
        raise ValueError(f"Cannot illustrate an invalid trace: {check}")
    metadata, splits = load_public(public_path)
    model = PeriodicModel.from_record(record["model"])
    trace = record["trace"]
    selected = np.array(trace["indices"])
    projections = np.array(
        [
            lift + sign * float(Fraction(phases[branch]))
            for phases, branch, sign, lift in zip(
                trace["phases"], trace["branches"], trace["signs"], trace["lifts"]
            )
        ]
    )
    np.testing.assert_allclose(
        splits["train"].inputs[selected] @ model.theta, projections, rtol=0, atol=1e-8
    )
    bound = max(1, int(np.ceil(np.max(np.abs(projections)))))
    choices = np.array(
        [
            (winding + sign * float(Fraction(phase)), row)
            for row, phases in enumerate(trace["phases"])
            for phase in phases
            for winding in range(-bound, bound + 1)
            for sign in (-1, 1)
            if -bound <= winding + sign * float(Fraction(phase)) <= bound
        ]
    )
    shown = list(selected) + [i for i in range(len(splits["train"].outputs)) if i not in selected]
    shown = np.array(shown[:256])
    specification = next(
        row for row in read_json(ROOT / "evidence/cases.json")["cases"] if row["id"] == case
    )
    return Story(
        case,
        public_path,
        result_path,
        record,
        model,
        splits["train"].inputs[shown],
        splits["train"].outputs[shown],
        shown,
        selected,
        projections,
        choices,
        bound,
        len(splits["train"].outputs),
        len(splits["validation"].outputs),
        metadata["B"],
        specification["provider"],
    )


class Illustration:
    def __init__(self, story):
        self.story = story
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "text.color": INK,
                "axes.labelcolor": INK,
                "xtick.color": SLATE,
                "ytick.color": SLATE,
                "axes.edgecolor": GRID,
                "figure.facecolor": PAPER,
                "axes.facecolor": PANEL,
                "savefig.facecolor": PAPER,
                "svg.fonttype": "none",
                "svg.hashsalt": "phaselattice-reconstruction-v1",
                "axes.unicode_minus": True,
            }
        )
        self.fig = plt.figure(figsize=(12, 6.75), dpi=160)
        self.fig.text(0.067, 0.946, "PHASELATTICE", fontsize=11, weight="bold", color=INDIGO)
        provider = (
            "Qiskit expectation response" if story.provider == "qiskit" else "Analytic response"
        )
        self.fig.text(
            0.935,
            0.946,
            f"{provider}  /  {len(story.model.weights)} input features",
            ha="right",
            fontsize=10,
            color=INDIGO,
        )
        self.fig.text(0.067, 0.865, "Periodic model reconstruction", fontsize=27, weight="bold")
        self.chapter_labels = []
        self.chapter_colors = (INDIGO, CORAL, TEAL)
        self.chapter_text_colors = (INDIGO, CORAL_TEXT, TEAL_TEXT)
        for x, title in [
            (0.067, "01  Observations"),
            (0.348, "02  Phase assignments"),
            (0.681, "03  Reconstructed model"),
        ]:
            self.chapter_labels.append(
                self.fig.text(
                    x,
                    0.789,
                    title,
                    fontsize=12,
                    color=SLATE,
                    bbox={
                        "boxstyle": "round,pad=0.55,rounding_size=0.35",
                        "facecolor": PAPER,
                        "edgecolor": "none",
                    },
                )
            )
        self.caption = self.fig.text(0.067, 0.721, "", fontsize=13, weight="medium", color=INK)

        self.left = self.fig.add_axes((0.083, 0.255, 0.473, 0.39))
        self.right = self.fig.add_axes((0.685, 0.255, 0.249, 0.39))
        self.left.set(xlim=(0, 1), ylim=(-1.06, 1.06), ylabel="Response")
        self.left.set_yticks([-1, -0.5, 0, 0.5, 1])
        self.left.set_xticks(np.linspace(0, 1, 5))
        self.left.axhline(0, color=GRID, lw=0.65, zorder=0)
        self.right.set(
            xlim=(-story.winding_bound - 0.18, story.winding_bound + 0.18),
            ylim=(len(story.selected) - 0.5, -0.5),
            yticks=range(len(story.selected)),
            yticklabels=range(1, len(story.selected) + 1),
            ylabel="Selected row",
            xlabel="Phase + integer cycles",
        )
        self.right.set_xticks([-story.winding_bound, 0, story.winding_bound])
        for ax in (self.left, self.right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0, pad=7, labelsize=10)
        self.left.set_title("Observed responses", loc="left", fontsize=11, color=INDIGO, pad=14)
        self.right.set_title("Candidate phases", loc="left", fontsize=11, color=CORAL_TEXT, pad=14)

        self.input_bound = max(1, int(np.ceil(np.max(np.abs(story.inputs[:, 0])))))
        self.initial = (story.inputs[:, 0] + self.input_bound) / (2 * self.input_bound)
        self.folded = (story.inputs @ story.model.theta) % 1
        self.points = self.left.scatter(
            self.initial, story.outputs, s=25, color=INDIGO, alpha=0.72, edgecolors="none", zorder=2
        )
        count = len(story.selected)
        self.selected_points = self.left.scatter(
            self.initial[:count],
            story.outputs[:count],
            s=62,
            facecolors=CORAL,
            edgecolors=PANEL,
            linewidths=1.0,
            alpha=0,
            zorder=4,
        )
        self.t = np.linspace(0, 1, 1200)
        self.wave = sum(
            k * np.cos(2 * np.pi * j * self.t) / story.model.denominator
            for j, k in enumerate(story.model.coefficients, 1)
        )
        (self.curve,) = self.left.plot([], [], color=TEAL, linewidth=2.8, zorder=3)
        self.candidates = self.right.scatter(
            story.choices[:, 0],
            story.choices[:, 1],
            s=18,
            color=INDIGO,
            alpha=0.32,
            edgecolors="none",
        )
        self.resolved = self.right.scatter(
            story.projections,
            np.arange(count),
            s=68,
            color=CORAL,
            alpha=0,
            edgecolors="none",
            zorder=3,
        )
        self.phase_note = self.fig.text(0.685, 0.13, "", fontsize=11, weight="medium", color=CORAL)
        self.equation = self.fig.text(0.083, 0.13, "", fontsize=11, weight="medium", color=TEAL)
        self.fig.text(
            0.067,
            0.060,
            "Saved Qiskit data" if story.provider == "qiskit" else "Saved analytic data",
            fontsize=9.5,
            color=SLATE,
        )
        self.fig.text(
            0.935,
            0.060,
            f"{len(story.inputs)} / {story.train_count:,} training points shown"
            f"  ·  {story.validation_count} validation points  ·  {story.precision}-bit fractional precision",
            ha="right",
            fontsize=9.5,
            color=SLATE,
        )
        self.frame(0)

    def frame(self, time):
        resolved = ease(time, CHAPTERS["joint_resolution"], 2.0)
        transformed = ease(time, CHAPTERS["coordinate_change"], 4.6)
        curve_fraction = ease(time, CHAPTERS["waveform"], CHAPTERS["result"])
        positions = (1 - transformed) * self.initial + transformed * self.folded
        self.points.set_offsets(np.column_stack((positions, self.story.outputs)))
        self.points.set_alpha(0.72 - 0.32 * transformed)
        count = len(self.story.selected)
        self.selected_points.set_offsets(
            np.column_stack((positions[:count], self.story.outputs[:count]))
        )
        self.selected_points.set_alpha(ease(time, 0.15, 0.75))
        self.candidates.set_alpha(0.32 - 0.23 * resolved)
        self.resolved.set_alpha(resolved)
        self.curve.set_data(
            self.t[: int(curve_fraction * len(self.t))],
            self.wave[: int(curve_fraction * len(self.t))],
        )

        if time < CHAPTERS["phase_choices"]:
            chapter, caption = 0, "Observed responses vs. first input coordinate"
        elif time < CHAPTERS["coordinate_change"]:
            chapter, caption = 1, f"Joint decoding of {count} phase assignments"
        elif time < CHAPTERS["waveform"]:
            chapter, caption = 2, "Mapping observations to the inferred phase"
        else:
            chapter, caption = 2, "Observed responses and reconstructed waveform"
        self.caption.set_text(caption)
        for i, label in enumerate(self.chapter_labels):
            color = self.chapter_colors[i]
            label.set_color(self.chapter_text_colors[i] if chapter == i else SLATE)
            label.set_weight("bold" if chapter == i else "normal")
            label.get_bbox_patch().set_facecolor(mix(PAPER, color, 0.10) if chapter == i else PAPER)
        initial_view = time < 3.4
        labels = [
            f"{v:g}".replace("-", "−") for v in np.linspace(-self.input_bound, self.input_bound, 5)
        ]
        self.left.set_xticklabels(labels if initial_view else ["0", "0.25", "0.5", "0.75", "1"])
        visibility = 1 - ease(time, 2.4, 2.65) + ease(time, 4.35, 4.6)
        for label in self.left.get_xticklabels():
            label.set_alpha(float(np.clip(visibility, 0, 1)))
        if time < CHAPTERS["coordinate_change"]:
            self.left.set_xlabel("First input coordinate")
        elif time < CHAPTERS["waveform"]:
            self.left.set_xlabel("Coordinate transformation")
        else:
            self.left.set_xlabel(r"Recovered phase · $(x\cdot\widehat{\theta})\;\mathrm{mod}\;1$")
        self.phase_note.set_text(f"{count} assignments · trace verified")
        self.phase_note.set_color(mix(PAPER, CORAL_TEXT, resolved))
        coefficients = ", ".join(str(k).replace("-", "−") for k in self.story.model.coefficients)
        self.equation.set_text(
            f"Recovered coefficients: ({coefficients}) / {self.story.model.denominator}"
        )
        self.equation.set_color(mix(PAPER, TEAL_TEXT, curve_fraction))

    def poster(self, destination):
        self.frame(POSTER_TIME)
        self.fig.savefig(
            destination / "reconstruction.png", dpi=160, metadata={"Software": "PhaseLattice"}
        )
        self.fig.savefig(
            destination / "reconstruction.svg", metadata={"Creator": "PhaseLattice", "Date": None}
        )


def ffmpeg(arguments):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True)


def render(story, destination, poster_only=False):
    destination.mkdir(parents=True, exist_ok=True)
    illustration = Illustration(story)
    illustration.poster(destination)
    if poster_only:
        plt.close(illustration.fig)
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Rendering the animation requires FFmpeg on PATH")
    with TemporaryDirectory(prefix="phaselattice-animation-") as temporary:
        master = Path(temporary) / "master.mkv"
        writer = FFMpegWriter(
            fps=VIDEO_FPS, codec="ffv1", extra_args=["-pix_fmt", "bgr0", "-threads", "2"]
        )
        with writer.saving(illustration.fig, master, dpi=160):
            for frame in range(SECONDS * VIDEO_FPS):
                illustration.frame(frame / VIDEO_FPS)
                writer.grab_frame()
                if frame % (4 * VIDEO_FPS) == 0:
                    print(f"Rendering {frame // VIDEO_FPS}/{SECONDS} seconds", flush=True)
        video = destination / "reconstruction.mp4"
        ffmpeg(
            [
                "-i",
                str(master),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-threads",
                "2",
                str(video),
            ]
        )
        palette = Path(temporary) / "palette.png"
        resize = f"fps={GIF_FPS},scale=1024:576:flags=lanczos"
        ffmpeg(
            [
                "-i",
                str(master),
                "-vf",
                f"{resize},palettegen=max_colors=256:stats_mode=full",
                "-frames:v",
                "1",
                "-update",
                "1",
                str(palette),
            ]
        )
        ffmpeg(
            [
                "-i",
                str(master),
                "-i",
                str(palette),
                "-lavfi",
                f"[0:v]{resize}[frames];[frames][1:v]paletteuse=dither=none:diff_mode=rectangle",
                "-loop",
                "0",
                str(destination / "reconstruction.gif"),
            ]
        )
    plt.close(illustration.fig)
    ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    manifest = dict(
        schema="phaselattice.animation/v1",
        case=story.case,
        generator_sha256=sha256(Path(__file__)),
        public_sha256=sha256(story.public_path),
        reconstruction_sha256=sha256(story.result_path),
        cohort_sha256=sha256(ROOT / "evidence/cases.json"),
        shown_training_rows=[int(row) for row in story.shown_rows],
        selected_training_rows=[int(row) for row in story.selected],
        seconds=SECONDS,
        video_fps=VIDEO_FPS,
        gif_fps=GIF_FPS,
        video_size=[1920, 1080],
        gif_size=[1024, 576],
        poster_time_seconds=POSTER_TIME,
        source="Saved public observations and a verified reconstruction trace; no hidden reference parameters",
        transition="Illustrative coordinate remapping, not optimizer iterations or solver runtime",
        phase_window=[-story.winding_bound, story.winding_bound],
        chapter_seconds=CHAPTERS,
        color_roles={
            "observations": INDIGO,
            "selected_assignments": CORAL,
            "recovered_curve": TEAL,
        },
        versions={name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        ffmpeg=ffmpeg_version,
        artifacts={
            path.name: sha256(path)
            for path in sorted(destination.iterdir())
            if path.stem == "reconstruction" and path.suffix in (".gif", ".mp4", ".png", ".svg")
        },
    )
    write_json(destination / "manifest.json", manifest)
    print("Wrote reconstruction.gif, reconstruction.mp4 and static PNG/SVG alternatives")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="quantum_d04_s03")
    parser.add_argument("--run", type=Path, default=ROOT / "evidence/run")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media")
    parser.add_argument("--poster-only", action="store_true")
    args = parser.parse_args()
    render(load_story(args.case, args.run), args.output, args.poster_only)


if __name__ == "__main__":
    main()
