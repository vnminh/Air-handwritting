"""Render trajectory + per-timestep gate-weight curves for qualitative examples.

Uses the existing selected K=2 checkpoint (local CPU forward pass only, no
training). Produces one figure per example: (a) the preprocessed trajectory
colored by time, (b) the three gate weight curves alpha_S(t), alpha_D(t),
alpha_F(t) predicted by the gated fusion module.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from airwriting.data import feature_branches, load_coordinates, preprocess_trajectory
from airwriting.metrics import greedy_decode
from airwriting.models import RecognitionModel

CHECKPOINT = ROOT / "models" / "best_fourier_k2.pt"

EXAMPLES = (
    (ROOT / "examples/bao_gio.csv", "bao giờ", "Correct", "bao_gio"),
    (ROOT / "examples/ngon_nui_cao_voi_voi.csv", "ngọn núi cao vời vợi", "Correct", "ngon_nui_cao_voi_voi"),
    (ROOT / "examples/bi_failure.csv", "bí", "Substitution", "bi_error"),
)

BRANCH_COLORS = {"spatial": "#3A6EA5", "dynamic": "#D97B29", "fourier": "#1E9E6B"}
BRANCH_LABELS = {"spatial": "Spatial", "dynamic": "Dynamic", "fourier": "Fourier"}


def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    tokens = list(checkpoint["tokens"])
    model = RecognitionModel(
        num_classes=len(tokens),
        representation=config["representation"],
        fusion=config["fusion"],
        backbone=config["backbone"],
        positional=config["positional"],
        fourier_scales=config["fourier_scales"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        branch_dropout=config.get("branch_dropout", 0.0),
        time_mask_probability=config.get("time_mask_probability", 0.0),
        time_mask_width=config.get("time_mask_width", 0),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, tokens


def draw_trajectory(axis, points: np.ndarray) -> None:
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments, cmap="viridis", norm=plt.Normalize(0, len(segments) - 1),
        linewidth=2.0, capstyle="round",
    )
    collection.set_array(np.arange(len(segments)))
    axis.add_collection(collection)
    axis.scatter(*points[0], s=34, color="#159447", edgecolor="white", linewidth=0.6, zorder=3)
    axis.scatter(*points[-1], s=34, color="#D64545", edgecolor="white", linewidth=0.6, zorder=3)
    axis.update_datalim(points)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


@torch.inference_mode()
def gate_weights_for(model, config, path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = load_coordinates(path)
    points = preprocess_trajectory(
        raw, length=config["length"], savgol=config["savgol"],
        spline=config["spline"], normalize=config["normalize"],
    )
    arrays = feature_branches(points, config["representation"], config["fourier_scales"])
    branches = {name: torch.from_numpy(values).unsqueeze(0) for name, values in arrays.items()}
    lengths = torch.tensor([len(points)])
    log_probs, _, weights = model(branches, lengths)
    if weights is None:
        raise ValueError("Checkpoint does not use gated fusion")
    weights = weights[0].cpu().numpy()  # (T, 3)
    return points, weights, model.fusion.names


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    model, config, tokens = load_model()
    output_dir = ROOT / "paper/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2, 3, figsize=(10.2, 4.6), gridspec_kw={"height_ratios": [1.1, 1.0]},
        constrained_layout=True,
    )
    for column, (path, reference, outcome, slug) in enumerate(EXAMPLES):
        points, weights, names = gate_weights_for(model, config, path)
        arrays = feature_branches(points, config["representation"], config["fourier_scales"])
        branch_tensor = {name: torch.from_numpy(v).unsqueeze(0) for name, v in arrays.items()}
        lengths = torch.tensor([len(points)])
        with torch.inference_mode():
            log_probs, _, _ = model(branch_tensor, lengths)
        prediction = greedy_decode(log_probs.float().cpu(), tokens)[0]

        top = axes[0, column]
        draw_trajectory(top, points)
        color = "#16845B" if outcome == "Correct" else "#B53A3A"
        top.set_title(f"{reference} $\\to$ {prediction}", fontsize=10.5, color=color, weight="bold")

        bottom = axes[1, column]
        t = np.arange(weights.shape[0])
        for index, name in enumerate(names):
            bottom.plot(t, weights[:, index], color=BRANCH_COLORS[name], linewidth=1.6, label=BRANCH_LABELS[name])
        bottom.set_ylim(0, 1)
        bottom.set_xlim(0, weights.shape[0] - 1)
        bottom.set_xlabel("Trajectory point", fontsize=9)
        if column == 0:
            bottom.set_ylabel("Gate weight", fontsize=9)
        bottom.tick_params(labelsize=8)
        bottom.grid(color="#E4E8EE", linewidth=0.6)
        if column == 2:
            bottom.legend(fontsize=8, loc="upper right", framealpha=0.9)

    output = output_dir / "gate_curves.png"
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
