"""Create a compact fixed-length cache so Modal does not stat/read 22k tiny files."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def load(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [(float(row["x"]), float(row["y"])) for row in csv.DictReader(handle)]
    return np.asarray(rows, dtype=np.float64)


def prepare(points, length=128):
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) >= 5:
        window = min(7, len(points) if len(points) % 2 else len(points) - 1)
        if window >= 5:
            points[:, 0] = savgol_filter(points[:, 0], window, 3, mode="interp")
            points[:, 1] = savgol_filter(points[:, 1], window, 3, mode="interp")
    if len(points) >= 4:
        base = np.linspace(0.0, 1.0, len(points))
        dense = np.linspace(0.0, 1.0, max(len(points), length))
        points = np.column_stack([CubicSpline(base, points[:, axis])(dense) for axis in range(2)])
    points -= points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 1e-12:
        points /= scale
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] <= 1e-12:
        return np.repeat(points[:1], length, axis=0).astype(np.float32)
    target = np.linspace(0, arc[-1], length)
    return np.column_stack([np.interp(target, arc, points[:, axis]) for axis in range(2)]).astype(np.float32)


if __name__ == "__main__":
    root = Path("Vni_air_writing/VNI_airwriting")
    manifest = json.loads(Path("manifests/split_seed_20260908.json").read_text(encoding="utf-8"))
    paths = [record["path"] for record in manifest["records"]]
    arrays = np.stack([prepare(load(root / path)) for path in paths])
    np.savez_compressed("preprocessed_128.npz", paths=np.asarray(paths), points=arrays)
    print(arrays.shape)
