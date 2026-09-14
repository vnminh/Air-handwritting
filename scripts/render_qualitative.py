"""Render qualitative inference examples with the paper preprocessing pipeline."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from airwriting.data import load_coordinates, preprocess_trajectory


EXAMPLES = (
    (ROOT / "examples/bao_gio.csv", "bao giờ", "bao giờ", "Correct", "bao_gio"),
    (
        ROOT / "examples/ngon_nui_cao_voi_voi.csv",
        "ngọn núi cao vời vợi",
        "ngọn núi cao vời vợi",
        "Correct",
        "ngon_nui_cao_voi_voi",
    ),
    (ROOT / "examples/bi_failure.csv", "bí", "bú", "Substitution", "bi_error"),
)


def draw_trajectory(axis, points: np.ndarray) -> None:
    """Draw a trajectory with color indicating temporal order."""
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=plt.Normalize(0, len(segments) - 1),
        linewidth=1.8,
        capstyle="round",
    )
    collection.set_array(np.arange(len(segments)))
    axis.add_collection(collection)
    axis.scatter(*points[0], s=38, color="#159447", edgecolor="white", linewidth=0.7, zorder=3)
    axis.scatter(*points[-1], s=38, color="#D64545", edgecolor="white", linewidth=0.7, zorder=3)
    axis.update_datalim(points)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="#D8DEE8", linewidth=0.55, alpha=0.75)
    axis.tick_params(labelsize=7, colors="#52637A")
    for spine in axis.spines.values():
        spine.set_color("#BFC9D8")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#17233C",
            "axes.labelcolor": "#52637A",
            "figure.facecolor": "white",
        }
    )
    output_dir = ROOT / "paper/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, reference, prediction, outcome, slug in EXAMPLES:
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
        figure.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05, wspace=0.04)
        raw = load_coordinates(path)
        normalized = preprocess_trajectory(raw, length=128, savgol=True, spline=True, normalize=True)
        draw_trajectory(axes[0], raw)
        draw_trajectory(axes[1], normalized)
        axes[0].set_title(f"Raw trajectory ({len(raw)} points)", fontsize=10, weight="bold", pad=7)
        axes[1].set_title("Preprocessed trajectory (128 points)", fontsize=10, weight="bold", pad=7)
        color = "#16845B" if outcome == "Correct" else "#B53A3A"
        figure.suptitle(
            f"Reference: {reference}\nPrediction: {prediction}   [{outcome}]",
            fontsize=11.5,
            weight="bold",
            color=color,
        )
        output = output_dir / f"qualitative_{slug}.png"
        figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
