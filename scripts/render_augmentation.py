"""Render exact training augmentation examples without ML dependencies."""
import bisect
import csv
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE = Path("Vni_air_writing/VNI_airwriting/2_grams/bao giờ_w/17.csv")
OUTPUT = Path("paper/figures/training_augmentation.png")
WIDTH, HEIGHT = 2200, 560
NAVY = "#17233C"
MUTED = "#5B6B80"
BLUE = "#4B7BEC"
GREEN = "#2A9D78"
ORANGE = "#DC8B18"
PURPLE = "#7656C6"
RED = "#D95D67"


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def load_points():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        points = [(float(row["x"]), float(row["y"])) for row in reader]
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    centered = [(x - cx, y - cy) for x, y in points]
    radius = max(math.hypot(x, y) for x, y in centered)
    return [(x / radius, y / radius) for x, y in centered]


def rotate(points, degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return [(c * x - s * y, s * x + c * y) for x, y in points]


def scale(points, sx, sy):
    return [(sx * x, sy * y) for x, y in points]


def noise(points, sigma, rng):
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in points]


def temporal_warp(points, sigma, rng):
    increments = [math.exp(rng.gauss(0, sigma)) for _ in range(len(points) - 1)]
    times = [0.0]
    for value in increments:
        times.append(times[-1] + value)
    times = [value / times[-1] for value in times]
    output = []
    for index in range(len(points)):
        target = index / (len(points) - 1)
        right = min(max(bisect.bisect_left(times, target), 1), len(times) - 1)
        left = right - 1
        ratio = (target - times[left]) / max(times[right] - times[left], 1e-12)
        x = points[left][0] + ratio * (points[right][0] - points[left][0])
        y = points[left][1] + ratio * (points[right][1] - points[left][1])
        output.append((x, y))
    return output


def draw_panel(canvas, index, title, subtitle, points, color):
    margin, gap = 50, 26
    panel_width = (WIDTH - 2 * margin - 4 * gap) / 5
    x0 = margin + index * (panel_width + gap)
    y0, x1, y1 = 125, x0 + panel_width, 505
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill="white", outline="#D9E1EC", width=3)
    draw.text(((x0 + x1) / 2, 30), title, font=font(27, True), fill=NAVY, anchor="ma")
    draw.text(((x0 + x1) / 2, 73), subtitle, font=font(19), fill=MUTED, anchor="ma")
    px0, py0, px1, py1 = x0 + 28, y0 + 28, x1 - 28, y1 - 28
    span_x = max(x for x, _ in points) - min(x for x, _ in points)
    span_y = max(y for _, y in points) - min(y for _, y in points)
    factor = 0.88 * min((px1 - px0) / max(span_x, 1e-9), (py1 - py0) / max(span_y, 1e-9))
    cx = (min(x for x, _ in points) + max(x for x, _ in points)) / 2
    cy = (min(y for _, y in points) + max(y for _, y in points)) / 2
    screen = [((px0 + px1) / 2 + (x - cx) * factor, (py0 + py1) / 2 - (y - cy) * factor) for x, y in points]
    draw.line(screen, fill=color, width=5, joint="curve")
    for point, fill in ((screen[0], GREEN), (screen[-1], RED)):
        x, y = point
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=fill, outline="white", width=2)


rng = random.Random(20260908)
base = load_points()
variants = [
    ("Original", "normalized trajectory", base, BLUE),
    ("Rotation", r"θ = +12°", rotate(base, 12), PURPLE),
    ("Stretching", "sx = 1.12, sy = 0.88", scale(base, 1.12, 0.88), ORANGE),
    ("Gaussian noise", "σ = 0.008", noise(base, 0.008, rng), RED),
    ("Time warping", "monotone, σ = 0.20", temporal_warp(base, 0.20, rng), GREEN),
]
image = Image.new("RGB", (WIDTH, HEIGHT), "#F7F9FC")
for idx, variant in enumerate(variants):
    draw_panel(image, idx, *variant)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT, quality=96)
print(OUTPUT)
