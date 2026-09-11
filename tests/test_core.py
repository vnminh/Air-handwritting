import json
from pathlib import Path

import numpy as np
import torch

from airwriting.data import PreprocessConfig, feature_branches, preprocess_trajectory
from airwriting.metrics import corpus_metrics, greedy_decode
from airwriting.models import RecognitionModel


def test_preprocessing_and_features():
    raw = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]], dtype=np.float32)
    points = preprocess_trajectory(raw, length=32)
    assert points.shape == (32, 2)
    assert np.isfinite(points).all()
    assert np.allclose(points.mean(axis=0), 0.0, atol=0.15)
    branches = feature_branches(points, "all", 4)
    assert {key: value.shape for key, value in branches.items()} == {
        "spatial": (32, 2), "dynamic": (32, 7), "fourier": (32, 16)
    }


def test_model_ctc_shape_and_relative_attention():
    model = RecognitionModel(
        12, representation="all", fusion="gated", backbone="conformer",
        positional="relative", fourier_scales=2, d_model=32, layers=2,
        heads=4, d_ff=64,
    )
    branches = {
        "spatial": torch.randn(3, 24, 2),
        "dynamic": torch.randn(3, 24, 7),
        "fourier": torch.randn(3, 24, 8),
    }
    output, lengths, weights = model(branches, torch.tensor([24, 24, 24]))
    assert output.shape == (24, 3, 12)
    assert lengths.tolist() == [24, 24, 24]
    assert weights.shape == (3, 24, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(3, 24), atol=1e-5)


def test_decoding_and_metrics():
    logits = torch.full((5, 1, 4), -10.0)
    for time, token in enumerate([0, 1, 1, 0, 2]):
        logits[time, 0, token] = 0.0
    assert greedy_decode(logits, ["<blank>", "a", "b", "c"]) == ["ab"]
    metrics = corpus_metrics(["abc"], ["adc"])
    assert metrics["cer"] == 1 / 3
    assert metrics["substitutions"] == 1


def test_manifest_has_disjoint_sources_and_complete_labels():
    path = Path("manifests/split_seed_20260908.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = {
        split: {record["source_id"] for record in manifest["records"] if record["split"] == split}
        for split in ("train", "val", "test")
    }
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    labels = {
        split: {record["label"] for record in manifest["records"] if record["split"] == split}
        for split in ids
    }
    assert all(len(value) == 660 for value in labels.values())
    assert manifest["statistics"] == {
        "labels": 660,
        "vocabulary_including_blank": 88,
        "train": 18202,
        "val": 2276,
        "test": 2276,
        "exact_duplicates_removed": 6,
    }

