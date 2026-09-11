from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import AirWritingDataset, PreprocessConfig, collate_batch
from .metrics import corpus_metrics, greedy_decode
from .models import RecognitionModel

TRAINING_VERSION = 2


@dataclass
class ExperimentConfig:
    name: str
    backbone: str = "conformer"
    representation: str = "all"
    fusion: str = "gated"
    positional: str = "relative"
    fourier_scales: int = 4
    length: int = 128
    savgol: bool = True
    spline: bool = True
    normalize: bool = True
    augment: bool = True
    seed: int = 42
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    batch_size: int = 512
    epochs: int = 40
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(branches, targets, lengths, target_lengths, device):
    return (
        {name: values.to(device, non_blocking=True) for name, values in branches.items()},
        targets.to(device, non_blocking=True),
        lengths.to(device, non_blocking=True),
        target_lengths.to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate(model, loader, tokens, device) -> tuple[dict, list[dict]]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict] = []
    for branches, _, lengths, _, labels, paths in loader:
        branches = {name: values.to(device, non_blocking=True) for name, values in branches.items()}
        lengths = lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            log_probs, _, _ = model(branches, lengths)
        predictions = greedy_decode(log_probs.float().cpu(), tokens)
        references.extend(labels)
        hypotheses.extend(predictions)
        rows.extend(
            {"path": path, "reference": reference, "prediction": prediction}
            for path, reference, prediction in zip(paths, labels, predictions)
        )
    return corpus_metrics(references, hypotheses), rows


def _loader(dataset, batch_size: int, shuffle: bool, workers: int = 0):
    generator = torch.Generator().manual_seed(dataset.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_batch,
        generator=generator,
    )


def run_experiment(
    config: ExperimentConfig,
    data_root: Path,
    manifest: dict,
    output_root: Path,
    point_cache: dict | None = None,
) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = output_root / config.name / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete" and result.get("training_version") == TRAINING_VERSION:
            print(f"Skipping completed run {config.name}/seed_{config.seed}", flush=True)
            return result

    preprocess = PreprocessConfig(config.length, config.savgol, config.spline, config.normalize)
    common = dict(
        data_root=data_root,
        manifest=manifest,
        representation=config.representation,
        fourier_scales=config.fourier_scales,
        preprocess=preprocess,
        seed=config.seed,
        point_cache=point_cache,
    )
    train_dataset = AirWritingDataset(split="train", augment=config.augment, **common)
    val_dataset = AirWritingDataset(split="val", augment=False, **common)
    test_dataset = AirWritingDataset(split="test", augment=False, **common)
    train_loader = _loader(train_dataset, config.batch_size, True)
    val_loader = _loader(val_dataset, config.batch_size, False)
    test_loader = _loader(test_dataset, config.batch_size, False)

    model = RecognitionModel(
        num_classes=len(manifest["vocabulary"]),
        representation=config.representation,
        fusion=config.fusion,
        backbone=config.backbone,
        positional=config.positional,
        fourier_scales=config.fourier_scales,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    warmup = max(1, config.epochs // 10)

    def learning_rate(epoch: int):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(config.epochs - warmup - 1, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_cer = float("inf")
    best_epoch = 0
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        running_loss = 0.0
        batches = 0
        for branches, targets, lengths, target_lengths, _, _ in train_loader:
            branches, targets, lengths, target_lengths = _to_device(
                branches, targets, lengths, target_lengths, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                log_probs, output_lengths, _ = model(branches, lengths)
                loss = loss_function(log_probs.float(), targets, output_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batches += 1
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, manifest["vocabulary"], device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
            "val_exact_accuracy": val_metrics["exact_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps({"run": config.name, **record}), flush=True)
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "config": asdict(config), "tokens": manifest["vocabulary"], "training_version": TRAINING_VERSION},
                run_dir / "best.pt",
            )
        elif epoch - best_epoch >= config.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics, val_predictions = evaluate(model, val_loader, manifest["vocabulary"], device)
    test_metrics, test_predictions = evaluate(model, test_loader, manifest["vocabulary"], device)
    (run_dir / "predictions_val.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in val_predictions) + "\n", encoding="utf-8"
    )
    (run_dir / "predictions_test.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in test_predictions) + "\n", encoding="utf-8"
    )
    result = {
        "status": "complete",
        "training_version": TRAINING_VERSION,
        "config": asdict(config),
        "parameters": parameters,
        "best_epoch": best_epoch,
        "duration_seconds": time.time() - started,
        "validation": val_metrics,
        "test": test_metrics,
        "history": history,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "cpu",
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@torch.inference_mode()
def run_robustness(
    checkpoint_path: Path,
    data_root: Path,
    manifest: dict,
    output_path: Path,
    point_cache: dict | None = None,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model = RecognitionModel(
        num_classes=len(checkpoint["tokens"]), representation=config.representation,
        fusion=config.fusion, backbone=config.backbone, positional=config.positional,
        fourier_scales=config.fourier_scales, d_model=config.d_model, layers=config.layers,
        heads=config.heads, d_ff=config.d_ff, dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    preprocess = PreprocessConfig(config.length, config.savgol, config.spline, config.normalize)
    settings = (
        [("rotation", value) for value in (0.0, 5.0, 10.0, 15.0, 20.0)]
        + [("noise", value) for value in (0.0, 0.005, 0.01, 0.02, 0.05)]
        + [("scale", value) for value in (0.8, 0.9, 1.0, 1.1, 1.2)]
        + [("time_warp", value) for value in (0.0, 0.1, 0.2, 0.35)]
    )
    results = []
    for kind, strength in settings:
        dataset = AirWritingDataset(
            data_root, manifest, "test", config.representation, config.fourier_scales,
            preprocess, False, config.seed, point_cache, (kind, strength),
        )
        loader = _loader(dataset, config.batch_size, False, workers=0)
        metrics, _ = evaluate(model, loader, checkpoint["tokens"], device)
        row = {"perturbation": kind, "strength": strength, **{k: metrics[k] for k in ("cer", "wer", "exact_accuracy")}}
        print(json.dumps(row), flush=True)
        results.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
