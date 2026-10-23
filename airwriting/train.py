"""Command-line entry point for training a model from scratch.

Example:

    python -m airwriting.train \\
        --data-root /path/to/VNI_airwriting \\
        --manifest manifests/split.json \\
        --output-root runs \\
        --name my_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .manifest import create_label_disjoint_manifest, create_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese air-writing recognizer from scratch."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory containing the VNI_airwriting label folders",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to a manifest JSON; generated automatically if it does not exist",
    )
    parser.add_argument(
        "--protocol",
        choices=("source", "lexical"),
        default="source",
        help="split protocol used only when --manifest does not already exist "
        "(source: same labels in every split; lexical: disjoint phrases)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="run")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backbone", default="conformer",
                         choices=("conformer", "transformer", "bilstm", "gru", "tcn"))
    parser.add_argument("--representation", default="all",
                         choices=("xy", "xy_delta", "dynamics", "fourier", "xy_dynamics",
                                  "xy_fourier", "dynamics_fourier", "all"))
    parser.add_argument("--fusion", default="gated",
                         choices=("concat", "sum", "static", "gated"))
    parser.add_argument("--positional", default="relative",
                         choices=("none", "absolute", "relative"))
    parser.add_argument("--fourier-scales", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--branch-dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        print(f"No manifest at {args.manifest}; generating a {args.protocol} split", flush=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        builder = create_manifest if args.protocol == "source" else create_label_disjoint_manifest
        manifest = builder(args.data_root, args.manifest, seed=args.seed)

    config = ExperimentConfig(
        name=args.name,
        seed=args.seed,
        backbone=args.backbone,
        representation=args.representation,
        fusion=args.fusion,
        positional=args.positional,
        fourier_scales=args.fourier_scales,
        length=args.length,
        augment=not args.no_augment,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        branch_dropout=args.branch_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    result = run_experiment(config, args.data_root, manifest, args.output_root)
    print(json.dumps({
        "run_dir": str(args.output_root / config.name / f"seed_{config.seed}"),
        "best_epoch": result["best_epoch"],
        "validation": result["validation"]["cer"],
        "test": result["test"]["cer"] if result["test"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
