"""Reusable inference API for Vietnamese air-writing trajectories."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .data import feature_branches, load_coordinates, preprocess_trajectory
from .metrics import greedy_decode
from .models import RecognitionModel


def default_checkpoint_path() -> Path:
    """Locate the paper checkpoint in a source checkout.

    Installed packages can use ``AIRWRITING_CHECKPOINT`` or pass an explicit
    path because model weights are distributed separately from Python wheels.
    """
    configured = os.environ.get("AIRWRITING_CHECKPOINT")
    if configured:
        return Path(configured).expanduser()
    package_directory = Path(__file__).resolve().parent
    candidates = (
        package_directory / "models" / "best_fourier_k2.pt",
        package_directory.parent / "models" / "best_fourier_k2.pt",
    )
    for repository_checkpoint in candidates:
        if repository_checkpoint.exists():
            return repository_checkpoint
    return Path("models/best_fourier_k2.pt")


def _select_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


class AirWritingRecognizer:
    """Load a checkpoint once and transcribe one or more trajectory CSV files."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path if checkpoint_path is not None else default_checkpoint_path()
        ).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Pass --checkpoint or set AIRWRITING_CHECKPOINT."
            )
        self.device = _select_device(device)
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        missing = {"model", "config", "tokens"} - set(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
        self.config = dict(checkpoint["config"])
        self.tokens = list(checkpoint["tokens"])
        self.model = RecognitionModel(
            num_classes=len(self.tokens),
            representation=self.config["representation"],
            fusion=self.config["fusion"],
            backbone=self.config["backbone"],
            positional=self.config["positional"],
            fourier_scales=self.config["fourier_scales"],
            d_model=self.config["d_model"],
            layers=self.config["layers"],
            heads=self.config["heads"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"],
            branch_dropout=self.config.get("branch_dropout", 0.0),
            time_mask_probability=self.config.get("time_mask_probability", 0.0),
            time_mask_width=self.config.get("time_mask_width", 0),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.inference_mode()
    def predict_points(self, points: np.ndarray, source: str = "<array>") -> dict:
        """Transcribe an ``(T, 2)`` array using checkpoint-defined processing."""
        raw_points = np.asarray(points)
        if raw_points.ndim != 2 or raw_points.shape[1] != 2:
            raise ValueError(f"Expected trajectory shape (T, 2), got {raw_points.shape}")
        prepared = preprocess_trajectory(
            raw_points,
            length=self.config["length"],
            savgol=self.config["savgol"],
            spline=self.config["spline"],
            normalize=self.config["normalize"],
        )
        arrays = feature_branches(
            prepared, self.config["representation"], self.config["fourier_scales"]
        )
        branches = {
            name: torch.from_numpy(values).unsqueeze(0).to(self.device)
            for name, values in arrays.items()
        }
        lengths = torch.tensor([len(prepared)], device=self.device)
        log_probs, _, weights = self.model(branches, lengths)
        result = {
            "source": source,
            "checkpoint": str(self.checkpoint_path),
            "prediction": greedy_decode(log_probs.float().cpu(), self.tokens)[0],
            "raw_points": int(len(raw_points)),
            "model_points": int(len(prepared)),
            "device": str(self.device),
        }
        if weights is not None:
            result["mean_fusion_weights"] = {
                name: float(value)
                for name, value in zip(
                    self.model.fusion.names, weights.mean((0, 1)).cpu()
                )
            }
        return result

    def predict_csv(self, csv_path: str | Path) -> dict:
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory CSV not found: {path}")
        return self.predict_points(load_coordinates(path), source=str(path))

    def predict_many(self, csv_paths: Iterable[str | Path]) -> list[dict]:
        return [self.predict_csv(path) for path in csv_paths]


def predict(
    csv_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> dict:
    """Convenience function for one-off inference."""
    return AirWritingRecognizer(checkpoint_path, device).predict_csv(csv_path)
