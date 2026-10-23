"""Benchmark runner: published baselines and spectral-representation study.

Usage (from the repository root):

    python scripts/bench.py run --phase compare --workers 3
    python scripts/bench.py summary

Each job trains one configuration on one protocol and seed in its own process
and writes ``benchmarks/<protocol>/<name>/seed_<seed>/result.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airwriting.data import PreprocessConfig, load_coordinates, preprocess_trajectory  # noqa: E402
from airwriting.experiment import ExperimentConfig, run_experiment  # noqa: E402

DATA_ROOT = ROOT / "Vni_air_writing" / "VNI_airwriting"
OUTPUT = ROOT / "benchmarks"
CACHE = OUTPUT / ".cache" / "points_128.pkl"
MANIFESTS = {
    "source": ROOT / "manifests" / "split_seed_20260908.json",
    "lexical": ROOT / "manifests" / "label_disjoint_seed_20260908.json",
}
SEEDS = (42, 3407, 2026)
NO_CLIP = 1e6

# Data pipeline shared by every method within a protocol: identical
# preprocessing, L=128, and the augmentation setting of our reference model.
PROTOCOL_DATA = {
    "source": dict(length=128, augment=True),
    "lexical": dict(length=128, augment=False),
}

# Our reference models, exactly as reported in the paper.
OURS = {
    "source": dict(
        representation="all", fusion="gated", positional="relative", fourier_scales=2,
        dropout=0.1, batch_size=512, epochs=40, patience=20,
    ),
    "lexical": dict(
        representation="dynamics_fourier", fusion="concat", positional="relative",
        fourier_scales=4, dropout=0.2, batch_size=256, epochs=70, patience=15,
    ),
}

# Published recipes (architecture + optimizer) from the original papers.
BASELINE_RECIPES = {
    # Carbune et al. 2020: point deltas, Adam 1e-4, batch 8, clip 9.
    "carbune_bilstm": dict(
        representation="delta", optimizer="adam", schedule="constant",
        learning_rate=1e-4, batch_size=8, grad_clip=9.0, epochs=100, patience=15,
    ),
    # Ott et al. 2022: raw signals, Adam 1e-4, batch 50.
    "ott_cnn_bilstm": dict(
        representation="xy", optimizer="adam", schedule="constant",
        learning_rate=1e-4, batch_size=50, grad_clip=NO_CLIP, epochs=200, patience=30,
    ),
    # Alam et al. 2020: coordinates, LSTM Adam 5e-4 / CNN Adam 1e-4, batch 256.
    "alam_lstm": dict(
        representation="xy", optimizer="adam", schedule="constant",
        learning_rate=5e-4, batch_size=256, grad_clip=NO_CLIP, epochs=100, patience=30,
    ),
    "alam_cnn": dict(
        representation="xy", optimizer="adam", schedule="constant",
        learning_rate=1e-4, batch_size=256, grad_clip=NO_CLIP, epochs=200, patience=30,
    ),
}


def _config(protocol: str, name: str, seed: int, **changes) -> ExperimentConfig:
    return ExperimentConfig(name=name, seed=seed, **{**PROTOCOL_DATA[protocol], **changes})


def phase_jobs(phase: str) -> list[tuple[str, ExperimentConfig]]:
    jobs: list[tuple[str, ExperimentConfig]] = []
    if phase == "compare":
        for seed in SEEDS:
            for protocol in ("source", "lexical"):
                jobs.append((protocol, _config(protocol, "ours", seed, **OURS[protocol])))
                for backbone, recipe in BASELINE_RECIPES.items():
                    jobs.append((protocol, _config(protocol, backbone, seed, backbone=backbone, **recipe)))
    elif phase == "spectral":
        # Screening at seed 42. Each protocol keeps its reference training
        # recipe and changes only the representation (and fusion for 'all').
        source = dict(OURS["source"], fusion="concat")
        lexical = dict(OURS["lexical"])
        variants = {
            "dynf": "dyn+fourier",
            "spec": "spec",
            "dyn_spec": "dyn+spec",
            "dynf_spec": "dyn+fourier+spec",
            "dynf_gspec": "dyn+fourier+gspec",
            "xy_spec": "xy+spec",
        }
        for name, representation in variants.items():
            jobs.append(("source", _config("source", f"spec_{name}", 42, **{**source, "representation": representation})))
            jobs.append(("lexical", _config("lexical", f"spec_{name}", 42, **{**lexical, "representation": representation})))
        jobs.append(("source", _config("source", "spec_all_gated", 42, **{**OURS["source"], "representation": "xy+dyn+fourier+spec"})))
        jobs.append(("lexical", _config("lexical", "spec_all_gated", 42, **{**lexical, "fusion": "gated", "representation": "xy+dyn+fourier+spec"})))
    elif phase == "transfer":
        # Do our representations also help the published architectures?
        for protocol in ("source", "lexical"):
            for backbone in ("carbune_bilstm", "ott_cnn_bilstm"):
                recipe = dict(BASELINE_RECIPES[backbone], representation="dyn+fourier", fourier_scales=OURS[protocol]["fourier_scales"])
                jobs.append((protocol, _config(protocol, f"{backbone}_dynf", 42, backbone=backbone, **recipe)))
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return jobs


def _result_path(protocol: str, config: ExperimentConfig) -> Path:
    return OUTPUT / protocol / config.name / f"seed_{config.seed}" / "result.json"


def _complete(protocol: str, config: ExperimentConfig) -> bool:
    path = _result_path(protocol, config)
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def _point_cache() -> dict:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    manifest = json.loads(MANIFESTS["source"].read_text(encoding="utf-8"))
    preprocess = PreprocessConfig(128, True, True, True)
    key = (preprocess.length, preprocess.savgol, preprocess.spline, preprocess.normalize)
    cache = {
        (record["path"], key): preprocess_trajectory(load_coordinates(DATA_ROOT / record["path"]), **preprocess.__dict__)
        for record in manifest["records"]
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(cache))
    return cache


def run_one(phase: str, protocol: str, name: str, seed: int) -> None:
    matches = [
        config for job_protocol, config in phase_jobs(phase)
        if job_protocol == protocol and config.name == name and config.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique job {phase}/{protocol}/{name}/{seed}")
    manifest = json.loads(MANIFESTS[protocol].read_text(encoding="utf-8"))
    run_experiment(matches[0], DATA_ROOT, manifest, OUTPUT / protocol, _point_cache())
    write_summary()


def run_phase(phase: str, workers: int) -> None:
    _point_cache()
    pending = [(p, c) for p, c in phase_jobs(phase) if not _complete(p, c)]
    print(f"{phase}: {len(pending)} pending jobs", flush=True)
    running: list[tuple[subprocess.Popen, str]] = []
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and len(running) < workers:
            protocol, config = pending.pop(0)
            tag = f"{protocol}/{config.name}/seed_{config.seed}"
            log = (logs / f"{protocol}_{config.name}_seed_{config.seed}.log").open("w")
            process = subprocess.Popen(
                [sys.executable, __file__, "one", "--phase", phase, "--protocol", protocol,
                 "--name", config.name, "--seed", str(config.seed)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((process, tag))
            print(f"started {tag}", flush=True)
        time.sleep(10)
        for process, tag in list(running):
            if process.poll() is not None:
                running.remove((process, tag))
                print(f"finished {tag} (exit {process.returncode})", flush=True)
    write_summary()


def write_summary() -> None:
    rows = []
    for path in sorted(OUTPUT.glob("*/*/seed_*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        config = result["config"]
        rows.append({
            "protocol": path.parts[-4],
            "name": config["name"],
            "seed": config["seed"],
            "backbone": config["backbone"],
            "representation": config["representation"],
            "fusion": config["fusion"],
            "parameters": result["parameters"],
            "best_epoch": result["best_epoch"],
            "minutes": round(result["duration_seconds"] / 60, 1),
            "val_cer": round(100 * result["validation"]["cer"], 2),
            "test_cer": round(100 * result["test"]["cer"], 2),
            "test_wer": round(100 * result["test"]["wer"], 2),
            "test_exact": round(100 * result["test"]["exact_accuracy"], 2),
        })
    if not rows:
        return
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["protocol"], row["name"]), []).append(row)
    lines = [
        "| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def stat(values: list[float]) -> str:
        mean = sum(values) / len(values)
        if len(values) == 1:
            return f"{mean:.2f}"
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        return f"{mean:.2f} ± {std:.2f}"

    for (protocol, name), group in sorted(groups.items()):
        lines.append(
            f"| {protocol} | {name} | {group[0]['representation']} | {group[0]['parameters']:,} "
            f"| {len(group)} | {stat([r['test_cer'] for r in group])} "
            f"| {stat([r['test_wer'] for r in group])} | {stat([r['test_exact'] for r in group])} |"
        )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--phase", required=True)
    run.add_argument("--workers", type=int, default=3)
    one = commands.add_parser("one")
    one.add_argument("--phase", required=True)
    one.add_argument("--protocol", required=True)
    one.add_argument("--name", required=True)
    one.add_argument("--seed", type=int, required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--phase", required=True)
    commands.add_parser("summary")
    args = parser.parse_args()
    if args.command == "run":
        run_phase(args.phase, args.workers)
    elif args.command == "one":
        run_one(args.phase, args.protocol, args.name, args.seed)
    elif args.command == "list":
        for protocol, config in phase_jobs(args.phase):
            print(protocol, json.dumps(asdict(config)))
    else:
        write_summary()


if __name__ == "__main__":
    main()
