from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from torch.utils.data import Dataset


BLANK = "<blank>"
SPACE = " "


def canonical_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def label_from_path(path: Path) -> str:
    name = path.parent.name
    if not name.endswith("_w"):
        raise ValueError(f"Expected label directory ending in '_w': {path}")
    return canonical_text(name[:-2])


def build_vocabulary(labels: Iterable[str]) -> list[str]:
    chars = sorted(set("".join(canonical_text(x) for x in labels)))
    return [BLANK, *chars]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_splits(seed: int) -> dict[str, str]:
    """Split repeated numeric source IDs globally while covering 10-shot labels.

    Every label has IDs 0..9. Some labels additionally have IDs 10..49. We
    partition each range 80/10/10, yielding 40/5/5 globally disjoint IDs and
    guaranteeing that every label occurs in train, validation, and test.
    """
    rng = random.Random(seed)
    common = list(range(10))
    extra = list(range(10, 50))
    rng.shuffle(common)
    rng.shuffle(extra)
    mapping: dict[str, str] = {}
    for value in common[:8] + extra[:32]:
        mapping[str(value)] = "train"
    for value in common[8:9] + extra[32:36]:
        mapping[str(value)] = "val"
    for value in common[9:] + extra[36:]:
        mapping[str(value)] = "test"
    return mapping


def create_manifest(data_root: Path, output_path: Path, seed: int = 20260908) -> dict:
    paths = sorted(data_root.rglob("*.csv"))
    source_split = _source_splits(seed)
    seen_hashes: dict[str, str] = {}
    records = []
    duplicates = []

    for path in paths:
        digest = _sha256(path)
        relative = path.relative_to(data_root).as_posix()
        if digest in seen_hashes:
            duplicates.append({"path": relative, "duplicate_of": seen_hashes[digest]})
            continue
        seen_hashes[digest] = relative
        source_id = path.stem
        if source_id not in source_split:
            raise ValueError(f"Unexpected source ID {source_id!r} in {relative}")
        records.append(
            {
                "path": relative,
                "label": label_from_path(path),
                "source_id": source_id,
                "split": source_split[source_id],
                "sha256": digest,
            }
        )

    labels = sorted({r["label"] for r in records})
    vocabulary = build_vocabulary(labels)
    counts = defaultdict(int)
    for record in records:
        counts[record["split"]] += 1

    payload = {
        "schema_version": 1,
        "seed": seed,
        "data_root": data_root.name,
        "split_unit": "numeric source_id shared globally across label directories",
        "writer_identity_confirmed": False,
        "records": records,
        "duplicates_removed": duplicates,
        "vocabulary": vocabulary,
        "statistics": {
            "labels": len(labels),
            "vocabulary_including_blank": len(vocabulary),
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "exact_duplicates_removed": len(duplicates),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_coordinates(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain x,y columns: {path}")
        for row in reader:
            rows.append((float(row["x"]), float(row["y"])))
    points = np.asarray(rows, dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"Invalid trajectory in {path}")
    return points


def _deduplicate_consecutive(points: np.ndarray) -> np.ndarray:
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    return points[keep]


def _arc_length_resample(points: np.ndarray, length: int) -> np.ndarray:
    points = _deduplicate_consecutive(points)
    if len(points) == 1:
        return np.repeat(points, length, axis=0)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segments)]
    if arc[-1] <= 1e-12:
        return np.repeat(points[:1], length, axis=0)
    target = np.linspace(0.0, arc[-1], length)
    return np.column_stack(
        [np.interp(target, arc, points[:, axis]) for axis in range(2)]
    )


def preprocess_trajectory(
    points: np.ndarray,
    length: int = 128,
    savgol: bool = True,
    spline: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).copy()
    points = _deduplicate_consecutive(points)

    if savgol and len(points) >= 5:
        window = min(7, len(points) if len(points) % 2 else len(points) - 1)
        if window >= 5:
            points[:, 0] = savgol_filter(points[:, 0], window, 3, mode="interp")
            points[:, 1] = savgol_filter(points[:, 1], window, 3, mode="interp")

    if spline and len(points) >= 4:
        base = np.linspace(0.0, 1.0, len(points))
        dense = np.linspace(0.0, 1.0, max(len(points), length))
        points = np.column_stack(
            [CubicSpline(base, points[:, axis])(dense) for axis in range(2)]
        )

    if normalize:
        points -= points.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 1e-12:
            points /= scale

    points = _arc_length_resample(points, length)
    return points.astype(np.float32)


def augment_trajectory(
    points: np.ndarray,
    rng: np.random.Generator,
    rotation_degrees: float = 15.0,
    scale_jitter: float = 0.15,
    noise_std: float = 0.008,
    time_warp_std: float = 0.20,
) -> np.ndarray:
    result = points.copy()
    angle = math.radians(float(rng.uniform(-rotation_degrees, rotation_degrees)))
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    result = result @ rotation.T
    result *= rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter, size=(1, 2)).astype(np.float32)
    result += rng.normal(0.0, noise_std, size=result.shape).astype(np.float32)

    # Smooth positive increments produce a monotone temporal warp.
    increments = np.exp(rng.normal(0.0, time_warp_std, size=len(result) - 1))
    warped_time = np.r_[0.0, np.cumsum(increments)]
    warped_time /= warped_time[-1]
    target = np.linspace(0.0, 1.0, len(result))
    result = np.column_stack(
        [np.interp(target, warped_time, result[:, axis]) for axis in range(2)]
    )
    return result.astype(np.float32)


def spatial_features(points: np.ndarray) -> np.ndarray:
    return points.astype(np.float32)


def dynamic_features(points: np.ndarray, mode: str = "full") -> np.ndarray:
    delta = np.diff(points, axis=0, prepend=points[:1])
    if mode == "delta":
        return delta.astype(np.float32)
    speed = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = delta / (speed + 1e-8)
    acceleration = np.diff(speed, axis=0, prepend=speed[:1])
    second = np.diff(delta, axis=0, prepend=delta[:1])
    numerator = delta[:, :1] * second[:, 1:] - delta[:, 1:] * second[:, :1]
    curvature = numerator / (np.power(speed, 3) + 1e-6)
    curvature = np.clip(curvature, -10.0, 10.0)
    return np.concatenate(
        [delta, speed, direction, acceleration, curvature], axis=1
    ).astype(np.float32)


def fourier_features(points: np.ndarray, scales: int = 4) -> np.ndarray:
    frequencies = (2.0 ** np.arange(scales, dtype=np.float32)) * np.pi
    phase = points[:, :, None] * frequencies[None, None, :]
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=2).reshape(len(points), -1).astype(np.float32)


def feature_branches(points: np.ndarray, representation: str, fourier_scales: int) -> dict[str, np.ndarray]:
    branches: dict[str, np.ndarray] = {}
    if representation in {"xy", "xy_delta", "xy_dynamics", "xy_fourier", "all"}:
        branches["spatial"] = spatial_features(points)
    if representation == "xy_delta":
        branches["dynamic"] = dynamic_features(points, "delta")
    elif representation in {"dynamics", "xy_dynamics", "dynamics_fourier", "all"}:
        branches["dynamic"] = dynamic_features(points, "full")
    if representation in {"fourier", "xy_fourier", "dynamics_fourier", "all"}:
        branches["fourier"] = fourier_features(points, fourier_scales)
    if not branches:
        raise ValueError(f"Unknown representation: {representation}")
    return branches


@dataclass(frozen=True)
class PreprocessConfig:
    length: int = 128
    savgol: bool = True
    spline: bool = True
    normalize: bool = True


class AirWritingDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest: dict,
        split: str,
        representation: str,
        fourier_scales: int = 4,
        preprocess: PreprocessConfig = PreprocessConfig(),
        augment: bool = False,
        seed: int = 42,
        point_cache: dict | None = None,
        perturbation: tuple[str, float] | None = None,
        augmentation_copies: int = 1,
        augmentation_rotation: float = 15.0,
        augmentation_scale: float = 0.15,
        augmentation_noise: float = 0.008,
        augmentation_time_warp: float = 0.20,
    ) -> None:
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.tokens = manifest["vocabulary"]
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.representation = representation
        self.fourier_scales = fourier_scales
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.perturbation = perturbation
        self.augmentation_copies = max(1, int(augmentation_copies))
        self.augmentation_rotation = float(augmentation_rotation)
        self.augmentation_scale = float(augmentation_scale)
        self.augmentation_noise = float(augmentation_noise)
        self.augmentation_time_warp = float(augmentation_time_warp)
        point_cache = point_cache if point_cache is not None else {}
        config_key = (preprocess.length, preprocess.savgol, preprocess.spline, preprocess.normalize)
        prepared = []
        for record in self.records:
            cache_key = (record["path"], config_key)
            if cache_key not in point_cache:
                point_cache[cache_key] = preprocess_trajectory(
                    load_coordinates(data_root / record["path"]), **preprocess.__dict__
                )
            prepared.append(point_cache[cache_key])
        self.points = np.stack(prepared)
        # Precompute a deterministic augmentation bank.  With more than one
        # copy, set_epoch rotates the training view instead of reusing the same
        # transformed trajectory throughout training.
        if self.augment:
            copies = []
            for copy_index in range(self.augmentation_copies):
                transformed = np.stack(
                    [
                        augment_trajectory(
                            points,
                            np.random.default_rng(
                                self.seed + copy_index * 1_000_003 + sample_index
                            ),
                            rotation_degrees=self.augmentation_rotation,
                            scale_jitter=self.augmentation_scale,
                            noise_std=self.augmentation_noise,
                            time_warp_std=self.augmentation_time_warp,
                        )
                        for sample_index, points in enumerate(self.points)
                    ]
                )
                copies.append(
                    [
                        feature_branches(points, self.representation, self.fourier_scales)
                        for points in transformed
                    ]
                )
            self.branch_arrays = {
                name: np.stack(
                    [
                        np.stack([features[name] for features in copy]).astype(np.float32)
                        for copy in copies
                    ]
                )
                for name in copies[0][0]
            }
        else:
            materialized_points = self.points.copy()
            if self.perturbation is not None:
                materialized_points = np.stack(
                    [self._apply_perturbation(points, i) for i, points in enumerate(materialized_points)]
                )
            feature_sets = [
                feature_branches(points, self.representation, self.fourier_scales)
                for points in materialized_points
            ]
            self.branch_arrays = {
                name: np.stack([features[name] for features in feature_sets]).astype(np.float32)
                for name in feature_sets[0]
            }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _apply_perturbation(self, points: np.ndarray, index: int) -> np.ndarray:
        kind, strength = self.perturbation
        rng = np.random.default_rng(self.seed + index)
        if kind == "rotation":
            angle = math.radians(strength)
            rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
            return points @ rotation.T
        if kind == "noise":
            return points + rng.normal(0.0, strength, points.shape).astype(np.float32)
        if kind == "scale":
            return points * strength
        if kind == "time_warp":
            increments = np.exp(rng.normal(0.0, strength, size=len(points) - 1))
            warped_time = np.r_[0.0, np.cumsum(increments)]
            warped_time /= warped_time[-1]
            target_time = np.linspace(0.0, 1.0, len(points))
            return np.column_stack([np.interp(target_time, warped_time, points[:, axis]) for axis in range(2)]).astype(np.float32)
        raise ValueError(f"Unknown perturbation: {kind}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        points = self.points[index]
        if self.augment:
            copy_index = max(0, self.epoch - 1) % self.augmentation_copies
            branches = {
                key: torch.from_numpy(value[copy_index, index])
                for key, value in self.branch_arrays.items()
            }
        else:
            branches = {
                key: torch.from_numpy(value[index])
                for key, value in self.branch_arrays.items()
            }
        target = torch.tensor([self.token_to_id[c] for c in record["label"]], dtype=torch.long)
        return branches, target, record["label"], record["path"]


def collate_batch(batch):
    branch_names = batch[0][0].keys()
    branches = {name: torch.stack([sample[0][name] for sample in batch]) for name in branch_names}
    targets = torch.cat([sample[1] for sample in batch])
    target_lengths = torch.tensor([len(sample[1]) for sample in batch], dtype=torch.long)
    input_lengths = torch.full((len(batch),), next(iter(branches.values())).shape[1], dtype=torch.long)
    labels = [sample[2] for sample in batch]
    paths = [sample[3] for sample in batch]
    return branches, targets, input_lengths, target_lengths, labels, paths
