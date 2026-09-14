"""Render the selected model as a publication PDF and high-resolution PNG."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-airwriting")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PNG = ROOT / "paper/figures/best_model_architecture.png"
PDF = ROOT / "paper/figures/best_model_architecture.pdf"

NAVY = "#17324D"
TEXT = "#26384A"
MUTED = "#52677C"
LINE = "#526B82"
LIGHT_LINE = "#A9B8C6"
SHADOW = "#DCE3EA"
WHITE = "#FFFFFF"
BOX_TEXT_BOUNDS = []


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    *,
    title_size: float = 10,
    body_size: float = 8.5,
    radius: float = 0.10,
) -> None:
    header_height = 0.70 if "\n" in title else 0.48
    axis.add_patch(
        FancyBboxPatch(
            (x + 0.05, y - 0.06),
            width,
            height,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=0,
            facecolor=SHADOW,
            alpha=0.55,
            zorder=1,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    axis.plot(
        [x + 0.08, x + width - 0.08],
        [y + height - header_height] * 2,
        color=edge,
        lw=1.1,
        zorder=3,
    )
    title_artist = axis.text(
        x + width / 2,
        y + height - header_height / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=NAVY,
        zorder=4,
    )
    body_artist = axis.text(
        x + width / 2,
        y + (height - header_height) / 2,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=TEXT,
        linespacing=1.25,
        zorder=4,
    )
    bounds = (x, y, x + width, y + height)
    BOX_TEXT_BOUNDS.extend(((title_artist, bounds, title), (body_artist, bounds, body)))


def arrow(axis, start, end, *, color=LINE, width=1.7, style="-") -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=width,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def orthogonal_arrow(axis, points, *, color=LINE, width=1.7) -> None:
    xs, ys = zip(*points)
    axis.plot(xs[:-1], ys[:-1], color=color, lw=width, solid_capstyle="round", zorder=4)
    arrow(axis, points[-2], points[-1], color=color, width=width)


def tensor_label(axis, x: float, y: float, text: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.22", "fc": WHITE, "ec": LIGHT_LINE, "lw": 0.8},
        zorder=8,
    )


def module(axis, x: float, width: float, title: str, subtitle: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, 0.69),
            width,
            0.82,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.3,
            edgecolor=color,
            facecolor=WHITE,
            zorder=3,
        )
    )
    title_artist = axis.text(x + width / 2, 1.20, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    body_artist = axis.text(x + width / 2, 0.91, subtitle, ha="center", va="center", fontsize=6.5, color=MUTED)
    bounds = (x, 0.69, x + width, 1.51)
    BOX_TEXT_BOUNDS.extend(((title_artist, bounds, title), (body_artist, bounds, subtitle)))


def validate_text_bounds(figure, axis) -> None:
    """Fail rendering if text extends beyond its containing module."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse = axis.transData.inverted()
    failures = []
    for artist, (left, bottom, right, top), label in BOX_TEXT_BOUNDS:
        extent = artist.get_window_extent(renderer=renderer).transformed(inverse)
        if extent.x0 < left or extent.x1 > right or extent.y0 < bottom or extent.y1 > top:
            failures.append(label.replace("\n", " / "))
    if failures:
        raise RuntimeError(f"Text exceeds its box: {failures}")


def main() -> None:
    BOX_TEXT_BOUNDS.clear()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(14, 7.5), facecolor=WHITE)
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 9.6)
    axis.axis("off")

    axis.text(0.30, 9.23, "SELECTED MODEL ARCHITECTURE", fontsize=9, fontweight="bold", color="#356C9A")
    axis.text(0.30, 8.78, "Multi-Representation Relative Conformer–CTC", fontsize=18, fontweight="bold", color=NAVY)
    axis.text(17.72, 9.16, "K = 2  •  d = 128  •  4 Conformer blocks  •  1.55M parameters", ha="right", fontsize=8.5, color=MUTED)
    axis.plot([0.30, 17.72], [8.46, 8.46], color="#D7E0E8", lw=1.4)

    box(axis, 0.25, 4.75, 1.45, 1.45, "Input", "Trajectory CSV\nordered $(x_t,y_t)$", "#EDF5FC", "#4D84B5")
    box(
        axis,
        2.15,
        4.10,
        2.05,
        2.75,
        "Preprocessing",
        "Remove duplicates\nSavitzky–Golay\nCubic interpolation\nCenter and scale\nArc-length resample",
        "#F4F6F8",
        "#71859A",
        body_size=8,
    )
    arrow(axis, (1.70, 5.48), (2.15, 5.48))
    tensor_label(axis, 1.92, 5.78, "$T\\times2$")

    branch_y = (6.45, 4.75, 3.05)
    branch_titles = ("Spatial branch", "Dynamic branch", "Fourier branch")
    branch_bodies = (
        "$S_t=[x_t,y_t]$\n$L\\times2$",
        "$\\Delta x,\\Delta y$, speed\nunit direction\nacceleration, curvature\n$L\\times7$",
        "$\\sin/\\cos(2^k\\pi p_t)$\n$k\\in\\{0,1\\}$; $K=2$\n$L\\times8$",
    )
    branch_faces = ("#EAF3FC", "#FFF4E5", "#F2ECFB")
    branch_edges = ("#397DB5", "#C77A18", "#7655A8")

    axis.text(5.78, 8.15, "FEATURE BRANCHES", ha="center", fontsize=7.5, fontweight="bold", color=MUTED)
    axis.text(8.11, 8.15, "BRANCH EMBEDDINGS", ha="center", fontsize=7.5, fontweight="bold", color=MUTED)
    axis.plot([4.72, 6.84], [8.03, 8.03], color=LIGHT_LINE, lw=0.9)
    axis.plot([7.18, 9.03], [8.03, 8.03], color=LIGHT_LINE, lw=0.9)

    axis.plot([4.20, 4.47], [5.48, 5.48], color=LINE, lw=1.7)
    axis.plot([4.47, 4.47], [3.75, 7.15], color=LINE, lw=1.7)
    for y, title, body, face, edge in zip(branch_y, branch_titles, branch_bodies, branch_faces, branch_edges):
        arrow(axis, (4.47, y + 0.70), (4.72, y + 0.70), color=edge)
        box(axis, 4.72, y, 2.12, 1.40, title, body, face, edge, title_size=9, body_size=7.5)
        arrow(axis, (6.84, y + 0.70), (7.18, y + 0.70), color=edge)
        box(
            axis,
            7.18,
            y + 0.11,
            1.85,
            1.18,
            "Embedding",
            "Linear $\\rightarrow$ LN $\\rightarrow$ SiLU\n$L\\times128$",
            "#F7F9FB",
            edge,
            title_size=9,
            body_size=7,
        )

    axis.plot([9.22, 9.22], [3.75, 7.15], color=LINE, lw=1.7)
    for y, edge in zip(branch_y, branch_edges):
        arrow(axis, (9.03, y + 0.70), (9.22, y + 0.70), color=edge)
    arrow(axis, (9.22, 5.47), (9.55, 5.47))

    box(
        axis,
        9.55,
        4.08,
        2.15,
        2.78,
        "Gated fusion",
        "$\\alpha_t=\\mathrm{softmax}($\n$W[h^S;h^D;h^F])$\n\n$h_t=\\sum_b\\alpha_t^b h_t^b$\n\nTime-dependent weights\n$L\\times128$",
        "#EAF7F2",
        "#248765",
        title_size=9.5,
        body_size=7.3,
    )
    arrow(axis, (11.70, 5.47), (12.05, 5.47))
    tensor_label(axis, 11.88, 5.80, "$L\\times128$")

    box(
        axis,
        12.05,
        3.83,
        2.35,
        3.28,
        "Relative Conformer\n×4",
        "Macaron FFN\nRelative MHSA: 4 heads\nClipped bias: $\\pm64$\nDepthwise Conv: $k=15$\nFFN width: 512\n$d_{model}=128$",
        "#EEF0FA",
        "#5269A9",
        title_size=8.5,
        body_size=7.4,
    )
    arrow(axis, (14.40, 5.47), (14.72, 5.47))
    tensor_label(axis, 14.55, 5.80, "$L\\times128$")

    box(
        axis,
        14.72,
        4.15,
        1.45,
        2.64,
        "CTC head",
        "Linear $128\\rightarrow88$\nLog-softmax\n\n86 characters\n+ space + blank",
        "#FFF5E8",
        "#BE7723",
        title_size=9.5,
        body_size=7.2,
    )
    arrow(axis, (16.17, 5.47), (16.48, 5.47))
    tensor_label(axis, 16.32, 5.80, "$L\\times88$")
    box(
        axis,
        16.48,
        4.35,
        1.27,
        2.24,
        "CTC decode",
        "Greedy path\nRemove blanks\nMerge repeats\n\nVietnamese\ntext",
        "#F4F6F8",
        "#667B8F",
        title_size=8,
        body_size=7,
    )

    container = FancyBboxPatch(
        (4.72, 0.22),
        9.68,
        1.95,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.25,
        linestyle="--",
        edgecolor="#8C9BAD",
        facecolor="#FAFBFC",
        zorder=1,
    )
    axis.add_patch(container)
    container_title = axis.text(4.98, 1.88, "CONFORMER BLOCK (×4)", fontsize=8, fontweight="bold", color="#5269A9")
    container_subtitle = axis.text(8.25, 1.88, "Macaron structure with residual connections", fontsize=7.2, color=MUTED)
    container_bounds = (4.72, 0.22, 14.40, 2.17)
    BOX_TEXT_BOUNDS.extend(
        (
            (container_title, container_bounds, "CONFORMER BLOCK (×4)"),
            (container_subtitle, container_bounds, "Macaron structure with residual connections"),
        )
    )

    modules = (
        (5.00, 1.25, "½ FFN", "LN • SiLU\nDropout", "#4D84B5"),
        (6.56, 1.70, "Relative MHSA", "$A_{ij}=Q_iK_j^T/\\sqrt{d_h}+b_{j-i}$", "#7655A8"),
        (8.58, 1.55, "Conv module", "GLU • DWConv\nBN • SiLU", "#248765"),
        (10.45, 1.25, "½ FFN", "LN • SiLU\nDropout", "#4D84B5"),
        (12.02, 1.40, "LayerNorm", "block output", "#BE7723"),
    )
    for x, width, title, subtitle, color in modules:
        module(axis, x, width, title, subtitle, color)
    for current, following in zip(modules, modules[1:]):
        arrow(axis, (current[0] + current[1], 1.10), (following[0], 1.10), width=1.3)
    axis.plot([4.94, 4.94, 13.47, 13.47], [0.65, 0.48, 0.48, 0.69], color="#8C9BAD", lw=1.1)
    axis.text(9.20, 0.34, "residual path", ha="center", va="center", fontsize=6.5, color=MUTED)
    orthogonal_arrow(axis, [(13.22, 3.83), (13.22, 2.48), (12.98, 2.17)], color="#8C9BAD", width=1.1)

    validate_text_bounds(figure, axis)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor=WHITE)
    figure.savefig(PDF, bbox_inches="tight", pad_inches=0.08, facecolor=WHITE)
    plt.close(figure)
    print(PNG.relative_to(ROOT))
    print(PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
