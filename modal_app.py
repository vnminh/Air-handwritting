from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import modal


app = modal.App("vni-airwriting-soict")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("airwriting", remote_path="/root/airwriting")
    .add_local_file(
        "manifests/split_seed_20260908.json",
        remote_path="/root/manifests/split_seed_20260908.json",
    )
    .add_local_file(
        "manifests/label_disjoint_seed_20260908.json",
        remote_path="/root/manifests/label_disjoint_seed_20260908.json",
    )
    .add_local_file(
        "models/best_fourier_k2.pt",
        remote_path="/root/models/best_fourier_k2.pt",
    )
)
data_volume = modal.Volume.from_name("vni-airwriting-data", create_if_missing=True)
results_volume = modal.Volume.from_name("vni-airwriting-results", create_if_missing=True)


def _base(name: str, **changes):
    from airwriting.experiment import ExperimentConfig

    return ExperimentConfig(name=name, **changes)


def experiment_suite(suite: str):
    if suite == "smoke":
        return [
            _base(
                "smoke", d_model=64, d_ff=128, layers=2, heads=4,
                batch_size=256, epochs=2, patience=2,
            )
        ]

    baseline = dict(representation="xy_dynamics", fusion="concat", augment=True)
    proposed = dict(representation="all", fusion="gated", positional="relative", augment=True)
    if suite == "core":
        configs = [
            _base("arch_bilstm", backbone="bilstm", positional="none", **baseline),
            _base("arch_gru", backbone="gru", positional="none", **baseline),
            # The dilated-convolution diagnostic is intentionally compact: on
            # the Modal L4 its unfused Conv1d kernels are much slower than the
            # sequence-attention models.  Keep it as an 8-epoch sanity check
            # rather than delaying the primary ablation matrix for hours.
            _base("arch_tcn", backbone="tcn", positional="none", d_model=64,
                  d_ff=128, layers=2, epochs=8, patience=4, **baseline),
            _base("arch_transformer", backbone="transformer", positional="absolute", **baseline),
            _base("baseline_conformer", backbone="conformer", positional="relative", **baseline),
            _base("repr_xy", representation="xy", fusion="concat", positional="relative"),
            _base("repr_xy_delta", representation="xy_delta", fusion="concat", positional="relative"),
            _base("repr_fourier", representation="fourier", fusion="concat", positional="relative"),
            _base("repr_xy_fourier", representation="xy_fourier", fusion="concat", positional="relative"),
            _base("repr_dynamics_fourier", representation="dynamics_fourier", fusion="concat", positional="relative"),
            _base("fusion_concat", representation="all", fusion="concat", positional="relative"),
            _base("fusion_sum", representation="all", fusion="sum", positional="relative"),
            _base("fusion_static", representation="all", fusion="static", positional="relative"),
            _base("proposed_gated", **proposed),
            _base("pe_none", representation="all", fusion="gated", positional="none"),
            _base("pe_absolute", representation="all", fusion="gated", positional="absolute"),
            _base("augmentation_none", representation="all", fusion="gated", positional="relative", augment=False),
        ]
        # Confirm the principal comparison over three independent seeds.
        for seed in (43, 44):
            configs.extend(
                [
                    replace(configs[0], seed=seed),
                    replace(configs[3], seed=seed),
                    replace(configs[4], seed=seed),
                    replace(configs[13], seed=seed),
                ]
            )
        return configs

    if suite == "repr_completion":
        # Completes Table II: the dynamics-only (D) row, now that
        # branch_dimensions/feature_branches support a pure "dynamics"
        # representation. S+D+F is already covered by fusion_concat above.
        return [_base("repr_dynamics", representation="dynamics", fusion="concat", positional="relative")]

    if suite == "priority":
        return [
            _base("arch_transformer", backbone="transformer", positional="absolute", **baseline),
            _base("baseline_conformer", backbone="conformer", positional="relative", **baseline),
            _base("proposed_gated", **proposed),
        ]

    if suite == "extended":
        configs = []
        for scales in (2, 4, 6, 8):
            configs.append(_base(f"fourier_k{scales}", fourier_scales=scales, **proposed))
        configs.extend(
            [
                _base("preprocess_raw", savgol=False, spline=False, normalize=False, **proposed),
                _base("preprocess_savgol", savgol=True, spline=False, normalize=True, **proposed),
                _base("preprocess_spline", savgol=False, spline=True, normalize=True, **proposed),
                _base("preprocess_full", savgol=True, spline=True, normalize=True, **proposed),
            ]
        )
        for length in (64, 96, 128, 192, 256):
            configs.append(_base(f"length_{length}", length=length, **proposed))
        return configs
    raise ValueError(f"Unknown suite: {suite}")


def lexical_ablation_suite() -> list[dict]:
    """Focused lexical-disjoint regularization ablation.

    Motivation: on the lexical-disjoint split (unseen test phrases), the raw
    spatial-coordinate branch lets the model shortcut toward memorizing
    absolute phrase shape rather than composing characters from motion, so
    dropping it (dynamics+Fourier, concat fusion) generalizes better than the
    full gated architecture even before adding any extra regularization. This
    suite holds the core architecture fixed (Conformer backbone, relative
    positional bias, CTC) and varies only (a) representation/fusion choice
    and (b) dropout strength, to separate the two effects instead of a blind
    hyperparameter sweep.
    """
    common = dict(
        positional="relative",
        fourier_scales=4,
        augment=False,
        batch_size=256,
        epochs=70,
        patience=15,
        learning_rate=1e-3,
    )

    def candidate(name: str, **changes) -> dict:
        return {"name": name, **common, **changes}

    return [
        # Dropout sweep around the known-good point (dynamics+Fourier, concat).
        candidate("lex_dynf_d20", representation="dynamics_fourier", fusion="concat", dropout=0.20),
        candidate("lex_dynf_d30", representation="dynamics_fourier", fusion="concat", dropout=0.30),
        candidate("lex_dynf_d40", representation="dynamics_fourier", fusion="concat", dropout=0.40),
        # Control: does the same dropout help the full gated architecture, or
        # is dropping the spatial branch what actually matters?
        candidate("lex_all_gated_d30", representation="all", fusion="gated", dropout=0.30),
        # Does stacking branch dropout on top of the best dropout help further?
        candidate(
            "lex_dynf_d30_branchdrop20", representation="dynamics_fourier",
            fusion="concat", dropout=0.30, branch_dropout=0.20,
        ),
    ]


def lexical_seed_suite(name: str) -> list[dict]:
    """Reruns one winning lexical-ablation config at two extra seeds for mean +/- std."""
    (candidate,) = [c for c in lexical_ablation_suite() if c["name"] == name]
    return [{**candidate, "seed": seed} for seed in (3407, 2026)]



def _write_summary(output_root: Path) -> None:
    results = []
    for path in output_root.glob("*/seed_*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            results.append(payload)
    rows = []
    for result in results:
        rows.append(
            {
                "name": result["config"]["name"],
                "seed": result["config"]["seed"],
                "backbone": result["config"]["backbone"],
                "representation": result["config"]["representation"],
                "fusion": result["config"]["fusion"],
                "positional": result["config"]["positional"],
                "dropout": result["config"].get("dropout", 0.1),
                "branch_dropout": result["config"].get("branch_dropout", 0.0),
                "time_mask_probability": result["config"].get("time_mask_probability", 0.0),
                "time_mask_width": result["config"].get("time_mask_width", 0),
                "confidence_penalty": result["config"].get("confidence_penalty", 0.0),
                "rdrop_alpha": result["config"].get("rdrop_alpha", 0.0),
                "augment": result["config"].get("augment", False),
                "augmentation_copies": result["config"].get("augmentation_copies", 1),
                "length": result["config"].get("length", 128),
                "fourier_scales": result["config"].get("fourier_scales", 4),
                "d_model": result["config"].get("d_model", 128),
                "layers": result["config"].get("layers", 4),
                "batch_size": result["config"].get("batch_size", 512),
                "learning_rate": result["config"].get("learning_rate", 1e-3),
                "weight_decay": result["config"].get("weight_decay", 1e-2),
                "parameters": result["parameters"],
                "best_epoch": result["best_epoch"],
                "val_cer": result["validation"]["cer"],
                "test_cer": result["test"]["cer"],
                "test_wer": result["test"]["wer"],
                "test_exact_accuracy": result["test"]["exact_accuracy"],
            }
        )
    rows.sort(key=lambda row: (row["name"], row["seed"]))
    (output_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    timeout=30 * 60,
    volumes={"/data": data_volume},
)
def prepare_data_volume() -> str:
    archive = Path("/data/staging/vni_airwriting.tar.gz")
    if not archive.exists():
        raise FileNotFoundError(f"Missing uploaded archive: {archive}")
    target = Path("/data")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(target)
    archive.unlink()
    data_volume.commit()
    return str(Path("/data/VNI_airwriting"))


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume},
)
def prepare_exact_cache() -> str:
    import numpy as np
    from airwriting.data import load_coordinates, preprocess_trajectory

    manifest = json.loads(Path("/root/manifests/split_seed_20260908.json").read_text(encoding="utf-8"))
    data_root = Path("/data/VNI_airwriting")
    paths = [record["path"] for record in manifest["records"]]
    points = np.stack(
        [preprocess_trajectory(load_coordinates(data_root / path), length=128, savgol=True, spline=True, normalize=True) for path in paths]
    )
    np.savez_compressed("/data/preprocessed_128.npz", paths=np.asarray(paths), points=points)
    data_volume.commit()
    return f"cached {len(paths)} trajectories"


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def train_suite(suite: str = "core") -> dict:
    from airwriting.experiment import run_experiment
    import numpy as np

    manifest = json.loads(
        Path("/root/manifests/split_seed_20260908.json").read_text(encoding="utf-8")
    )
    data_root = Path("/data/VNI_airwriting")
    if not data_root.exists():
        raise FileNotFoundError("Upload VNI_airwriting to the vni-airwriting-data volume first")
    output_root = Path("/results/experiments")
    point_cache: dict = {}
    compact_cache = Path("/data/preprocessed_128.npz")
    if compact_cache.exists():
        cached = np.load(compact_cache, allow_pickle=False)
        config_key = (128, True, True, True)
        point_cache.update({(str(path), config_key): points for path, points in zip(cached["paths"], cached["points"])})
        print(f"Loaded compact trajectory cache: {len(point_cache)} samples", flush=True)
    completed = []
    for config in experiment_suite(suite):
        print(f"Starting {config.name}/seed_{config.seed}", flush=True)
        result = run_experiment(config, data_root, manifest, output_root, point_cache)
        completed.append(
            {"name": config.name, "seed": config.seed, "test_cer": result["test"]["cer"]}
        )
        _write_summary(output_root)
        results_volume.commit()
    return {"suite": suite, "completed": completed}


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def train_lexical_candidate(config_payload: dict) -> dict:
    from airwriting.experiment import ExperimentConfig, run_experiment
    import numpy as np

    config = ExperimentConfig(**config_payload)
    manifest = json.loads(
        Path("/root/manifests/label_disjoint_seed_20260908.json").read_text(
            encoding="utf-8"
        )
    )
    data_root = Path("/data/VNI_airwriting")
    if not data_root.exists():
        raise FileNotFoundError("VNI_airwriting is missing from the Modal data volume")
    output_root = Path("/results/lexical_regularization")
    point_cache: dict = {}
    compact_cache = Path("/data/preprocessed_128.npz")
    if compact_cache.exists():
        cached = np.load(compact_cache, allow_pickle=False)
        config_key = (128, True, True, True)
        point_cache.update(
            {
                (str(path), config_key): points
                for path, points in zip(cached["paths"], cached["points"])
            }
        )
        print(f"Loaded trajectory cache: {len(point_cache)} samples", flush=True)
    result = run_experiment(config, data_root, manifest, output_root, point_cache)
    results_volume.commit()
    return {
        "name": config.name,
        "seed": config.seed,
        "best_epoch": result["best_epoch"],
        "val_cer": result["validation"]["cer"],
        "test_cer": result["test"]["cer"],
        "test_wer": result["test"]["wer"],
        "test_exact_accuracy": result["test"]["exact_accuracy"],
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=30 * 60,
    volumes={"/results": results_volume},
)
def summarize_lexical_runs() -> dict:
    results_volume.reload()
    output_root = Path("/results/lexical_regularization")
    _write_summary(output_root)
    results_volume.commit()
    rows = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    eligible = [row for row in rows if row["name"].startswith("lex_")]
    return min(eligible, key=lambda row: row["val_cer"])


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/results": results_volume},
)
def robustness() -> dict:
    from airwriting.experiment import run_robustness
    import numpy as np

    manifest = json.loads(Path("/root/manifests/split_seed_20260908.json").read_text(encoding="utf-8"))
    data_root = Path("/data/VNI_airwriting")
    output_root = Path("/results/experiments")
    point_cache: dict = {}
    compact_cache = Path("/data/preprocessed_128.npz")
    if compact_cache.exists():
        cached = np.load(compact_cache, allow_pickle=False)
        config_key = (128, True, True, True)
        point_cache.update({(str(path), config_key): points for path, points in zip(cached["paths"], cached["points"])})
        print(f"Loaded compact trajectory cache: {len(point_cache)} samples", flush=True)
    completed = []
    for name in ("arch_transformer", "baseline_conformer", "proposed_gated", "fourier_k2"):
        checkpoint = output_root / name / "seed_42" / "best.pt"
        output = output_root / name / "seed_42" / "robustness.json"
        if not checkpoint.exists():
            print(f"Skipping robustness for unavailable checkpoint: {name}", flush=True)
            continue
        run_robustness(checkpoint, data_root, manifest, output, point_cache)
        completed.append(name)
        results_volume.commit()
    return {"completed": completed}


@app.function(
    image=image,
    gpu="L4",
    timeout=10 * 60,
    volumes={"/data": data_volume},
)
def infer_example(relative_path: str) -> dict:
    """Run the paper checkpoint on one trajectory in the Modal data volume."""
    import json
    from airwriting.inference import AirWritingRecognizer

    recognizer = AirWritingRecognizer("/root/models/best_fourier_k2.pt")
    output = recognizer.predict_csv(Path("/data/VNI_airwriting") / relative_path)
    output["path"] = relative_path
    print(json.dumps(output, ensure_ascii=False), flush=True)
    return output


@app.local_entrypoint()
def main(suite: str = "core"):
    if suite == "robustness":
        print(robustness.remote())
    elif suite == "lexical_smoke":
        print(
            json.dumps(
                train_lexical_candidate.remote(
                    {
                        "name": "lex_smoke",
                        "d_model": 32,
                        "d_ff": 64,
                        "layers": 1,
                        "heads": 4,
                        "batch_size": 512,
                        "epochs": 1,
                        "patience": 1,
                        "dropout": 0.2,
                        "branch_dropout": 0.2,
                        "time_mask_probability": 0.5,
                        "time_mask_width": 8,
                        "augment": True,
                        "augmentation_copies": 2,
                        "augmentation_rotation": 5.0,
                        "augmentation_scale": 0.05,
                        "augmentation_noise": 0.003,
                        "augmentation_time_warp": 0.1,
                    }
                ),
                indent=2,
            )
        )
    elif suite == "lexical_ablation":
        payloads = lexical_ablation_suite()
        completed = list(train_lexical_candidate.map(payloads))
        print(json.dumps({"completed": completed}, indent=2))
        print(json.dumps({"validation_selected": summarize_lexical_runs.remote()}, indent=2))
    elif suite.startswith("lexical_seeds:"):
        name = suite.split(":", 1)[1]
        payloads = lexical_seed_suite(name)
        completed = list(train_lexical_candidate.map(payloads))
        print(json.dumps({"completed": completed}, indent=2))
    else:
        print(train_suite.remote(suite))
