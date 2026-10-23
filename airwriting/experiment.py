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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import AirWritingDataset, PreprocessConfig, collate_batch
from .metrics import corpus_metrics, greedy_decode
from .baselines import BASELINES, BaselineModel
from .models import RecognitionModel, branch_dimensions

TRAINING_VERSION = 4


@dataclass
class ExperimentConfig:
    name: str
    experiment_role: str = ""
    hypothesis: str = ""
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
    branch_dropout: float = 0.0
    time_mask_probability: float = 0.0
    time_mask_width: int = 0
    confidence_penalty: float = 0.0
    rdrop_alpha: float = 0.0
    augmentation_copies: int = 1
    augmentation_rotation: float = 15.0
    augmentation_scale: float = 0.15
    augmentation_noise: float = 0.008
    augmentation_time_warp: float = 0.20
    batch_size: int = 512
    epochs: int = 40
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    schedule: str = "cosine"
    grad_clip: float = 5.0


def build_model(config: ExperimentConfig, num_classes: int, num_labels: int = 0) -> nn.Module:
    """Build the proposed recognizer or a re-implemented published baseline.

    CTC models predict ``num_classes`` symbols; closed-set classifiers predict
    one of ``num_labels`` training labels.
    """
    if config.backbone in BASELINES:
        widths = branch_dimensions(config.representation, config.fourier_scales)
        outputs = num_labels if config.backbone.startswith("alam") else num_classes
        return BaselineModel(config.backbone, widths, outputs, config.length)
    return RecognitionModel(
        num_classes=num_classes,
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
        branch_dropout=config.branch_dropout,
        time_mask_probability=config.time_mask_probability,
        time_mask_width=config.time_mask_width,
    )


def _is_classifier(model: nn.Module) -> bool:
    return getattr(model, "task", "ctc") == "classification"


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
def evaluate(model, loader, tokens, device, class_labels=None) -> tuple[dict, list[dict]]:
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict] = []
    for branches, _, lengths, _, labels, paths in loader:
        branches = {name: values.to(device, non_blocking=True) for name, values in branches.items()}
        lengths = lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(branches, lengths)
        if _is_classifier(model):
            predictions = [class_labels[index] for index in outputs.argmax(dim=-1).tolist()]
        else:
            predictions = greedy_decode(outputs[0].float().cpu(), tokens)
        references.extend(labels)
        hypotheses.extend(predictions)
        rows.extend(
            {"path": path, "reference": reference, "prediction": prediction}
            for path, reference, prediction in zip(paths, labels, predictions)
        )
    return corpus_metrics(references, hypotheses), rows


def _masked_frame_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    mask = positions < lengths.unsqueeze(0)
    return values.masked_select(mask).mean()


def _symmetric_frame_kl(
    first: torch.Tensor, second: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    first_probability = first.exp()
    second_probability = second.exp()
    first_to_second = (first_probability * (first - second)).sum(dim=-1)
    second_to_first = (second_probability * (second - first)).sum(dim=-1)
    return 0.5 * (
        _masked_frame_mean(first_to_second, lengths)
        + _masked_frame_mean(second_to_first, lengths)
    )


def _confidence_regularizer(log_probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    negative_entropy = (log_probs.exp() * log_probs).sum(dim=-1)
    return _masked_frame_mean(negative_entropy, lengths)


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
    evaluate_test: bool = True,
) -> dict:
    labels_by_split = {
        split: {
            record["label"] for record in manifest["records"]
            if record["split"] == split
        }
        for split in ("train", "val", "test")
    }
    train_characters = set("".join(labels_by_split["train"]))
    for split in ("val", "test"):
        missing = sorted(set("".join(labels_by_split[split])) - train_characters)
        if missing:
            raise ValueError(
                f"{split} contains output characters absent from train: {missing!r}"
            )
    if manifest.get("evaluation_protocol") == "lexical-disjoint open-phrase recognition":
        if labels_by_split["train"] & labels_by_split["val"]:
            raise ValueError("Lexical-disjoint manifest has train/validation label overlap")
        if labels_by_split["train"] & labels_by_split["test"]:
            raise ValueError("Lexical-disjoint manifest has train/test label overlap")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = output_root / config.name / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reusable_statuses = {"complete"} if evaluate_test else {"validated", "complete"}
        if (
            result.get("status") in reusable_statuses
            and result.get("training_version") == TRAINING_VERSION
        ):
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
    train_dataset = AirWritingDataset(
        split="train",
        augment=config.augment,
        augmentation_copies=config.augmentation_copies,
        augmentation_rotation=config.augmentation_rotation,
        augmentation_scale=config.augmentation_scale,
        augmentation_noise=config.augmentation_noise,
        augmentation_time_warp=config.augmentation_time_warp,
        **common,
    )
    val_dataset = AirWritingDataset(split="val", augment=False, **common)
    train_loader = _loader(train_dataset, config.batch_size, True)
    val_loader = _loader(val_dataset, config.batch_size, False)

    class_labels = sorted(labels_by_split["train"])
    label_to_class = {label: index for index, label in enumerate(class_labels)}
    model = build_model(config, len(manifest["vocabulary"]), len(class_labels)).to(device)
    classifier = _is_classifier(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    warmup = max(1, config.epochs // 10)

    def learning_rate(epoch: int):
        if config.schedule == "constant":
            return 1.0
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
        for branches, targets, lengths, target_lengths, batch_labels, _ in train_loader:
            branches, targets, lengths, target_lengths = _to_device(
                branches, targets, lengths, target_lengths, device
            )
            optimizer.zero_grad(set_to_none=True)
            if classifier:
                classes = torch.tensor(
                    [label_to_class[label] for label in batch_labels], device=device
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(branches, lengths)
                loss = F.cross_entropy(logits.float(), classes)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.item())
                batches += 1
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                log_probs, output_lengths, _ = model(branches, lengths)
                float_log_probs = log_probs.float()
                loss = loss_function(
                    float_log_probs, targets, output_lengths, target_lengths
                )
                if config.rdrop_alpha > 0.0:
                    second_log_probs, _, _ = model(branches, lengths)
                    second_float = second_log_probs.float()
                    second_ctc = loss_function(
                        second_float, targets, output_lengths, target_lengths
                    )
                    loss = 0.5 * (loss + second_ctc) + config.rdrop_alpha * _symmetric_frame_kl(
                        float_log_probs, second_float, output_lengths
                    )
                if config.confidence_penalty > 0.0:
                    loss = loss + config.confidence_penalty * _confidence_regularizer(
                        float_log_probs, output_lengths
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batches += 1
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, manifest["vocabulary"], device, class_labels)
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
                {"model": model.state_dict(), "config": asdict(config), "tokens": manifest["vocabulary"], "labels": class_labels, "training_version": TRAINING_VERSION},
                run_dir / "best.pt",
            )
        elif epoch - best_epoch >= config.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics, val_predictions = evaluate(model, val_loader, manifest["vocabulary"], device, class_labels)
    (run_dir / "predictions_val.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in val_predictions) + "\n", encoding="utf-8"
    )
    test_metrics = None
    if evaluate_test:
        test_dataset = AirWritingDataset(split="test", augment=False, **common)
        test_loader = _loader(test_dataset, config.batch_size, False)
        test_metrics, test_predictions = evaluate(
            model, test_loader, manifest["vocabulary"], device, class_labels
        )
        (run_dir / "predictions_test.jsonl").write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False) for row in test_predictions
            )
            + "\n",
            encoding="utf-8",
        )
    result = {
        "status": "complete" if evaluate_test else "validated",
        "training_version": TRAINING_VERSION,
        "config": asdict(config),
        "parameters": parameters,
        "best_epoch": best_epoch,
        "duration_seconds": time.time() - started,
        "validation": val_metrics,
        "test": test_metrics,
        "selection": {
            "criterion": "validation.cer",
            "test_locked_during_training": not evaluate_test,
        },
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
def evaluate_saved_checkpoint_on_test(
    checkpoint_path: Path,
    data_root: Path,
    manifest: dict,
    point_cache: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate a frozen checkpoint on test after validation-only selection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    class_labels = checkpoint.get("labels")
    model = build_model(config, len(checkpoint["tokens"]), len(class_labels or [])).to(device)
    model.load_state_dict(checkpoint["model"])
    preprocess = PreprocessConfig(
        config.length, config.savgol, config.spline, config.normalize
    )
    test_dataset = AirWritingDataset(
        data_root=data_root,
        manifest=manifest,
        split="test",
        representation=config.representation,
        fourier_scales=config.fourier_scales,
        preprocess=preprocess,
        augment=False,
        seed=config.seed,
        point_cache=point_cache,
    )
    test_loader = _loader(test_dataset, config.batch_size, False)
    return evaluate(model, test_loader, checkpoint["tokens"], device, class_labels)


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
    class_labels = checkpoint.get("labels")
    model = build_model(config, len(checkpoint["tokens"]), len(class_labels or [])).to(device)
    model.load_state_dict(checkpoint["model"])
    preprocess = PreprocessConfig(config.length, config.savgol, config.spline, config.normalize)
    settings = (
        [("rotation", value) for value in (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)]
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
        metrics, _ = evaluate(model, loader, checkpoint["tokens"], device, class_labels)
        row = {"perturbation": kind, "strength": strength, **{k: metrics[k] for k in ("cer", "wer", "exact_accuracy")}}
        print(json.dumps(row), flush=True)
        results.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
