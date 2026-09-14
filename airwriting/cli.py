"""Command-line interface for trajectory transcription."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import AirWritingRecognizer, default_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe Vietnamese air-writing trajectory CSV files."
    )
    parser.add_argument(
        "csv", nargs="+", type=Path, help="CSV file(s) containing x and y columns"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="checkpoint path (default: repository paper checkpoint)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="inference device",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="emit one compact JSON object per input file",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    recognizer = AirWritingRecognizer(args.checkpoint, args.device)
    results = recognizer.predict_many(args.csv)
    if args.jsonl:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
    else:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
