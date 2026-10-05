"""Command-line interface for saved-observation reconstruction."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import Reconstruction, reconstruct
from .family import compile_family, default_family


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="phaselattice", description="Reconstruct a periodic response model from observations."
    )
    root.add_argument("--version", action="version", version=f"PhaseLattice {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the blinded reconstruction demonstration")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--provider", choices=("analytic", "qiskit"), default="qiskit")
    recover = commands.add_parser("recover", help="Reconstruct from a public JSON or JSON.gz file")
    recover.add_argument("observations", type=Path)
    recover.add_argument("--family", type=Path)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--no-screen", action="store_true")
    recover.add_argument("--relations", choices=("full", "single"), default="full")
    verify = commands.add_parser("verify", help="Replay a reconstruction's consistency trace")
    verify.add_argument("observations", type=Path)
    verify.add_argument("reconstruction", type=Path)
    export = commands.add_parser("export", help="Write a standalone NumPy/PyTorch/Qiskit model")
    export.add_argument("reconstruction", type=Path)
    export.add_argument("--output", type=Path, required=True)
    family = commands.add_parser("family", help="Compile a finite waveform family")
    family.add_argument("--degree", type=int, default=3)
    family.add_argument("--denominator", type=int, default=4)
    family.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser(
        "generate", help="Create observations and separate evaluator files"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--dimension", type=int, default=4)
    generate.add_argument("--seed", type=int, default=410001)
    generate.add_argument("--provider", choices=("analytic", "qiskit"), default="analytic")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "recover":
            result = reconstruct(
                args.observations,
                family=args.family,
                screen=not args.no_screen,
                relations=args.relations,
            )
            result.save(args.output)
            print(f"{result.record['status']}  {result.seconds:.3f} s  {args.output}")
            return 0 if result.recovered else 3
        if args.command == "verify":
            check = Reconstruction.load(args.reconstruction).verify(args.observations)
            print(json.dumps(check, indent=2))
            return 0 if check["valid"] else 1
        if args.command == "export":
            model = Reconstruction.load(args.reconstruction).model
            if model is None:
                raise ValueError("An unresolved reconstruction has no model to export")
            model.export(args.output)
            print(args.output)
        elif args.command == "family":
            family = compile_family(args.output, args.degree, args.denominator)
            print(
                f"{family['census']['covered']} waveforms; {len(family['bands'])} bands; {args.output}"
            )
        elif args.command == "generate":
            from .synthetic import generate

            if any(args.output.glob("*.json")):
                raise ValueError("Choose an empty output directory")
            generate(
                args.output, default_family(), args.dimension, args.seed, provider=args.provider
            )
            print(args.output / "observations.json")
        else:
            from .demo import run_demo

            metrics = run_demo(args.output, args.provider)
            print(f"Recovered waveform and weights in {metrics['reconstruction_seconds']:.3f} s")
            print(f"Test MSE / variance: {metrics['loss_over_variance']:.3e}")
            print(
                f"Shifted-input gradient error: {metrics['shifted_inputs']['gradient_relative_rms_error']:.3e}"
            )
            print(f"Artifacts: {args.output.resolve()}")
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"phaselattice: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
