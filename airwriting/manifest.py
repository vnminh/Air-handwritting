from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BLANK = "<blank>"


def canonical_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def label_from_path(path: Path) -> str:
    name = path.parent.name
    if not name.endswith("_w"):
        raise ValueError(f"Expected label directory ending in '_w': {path}")
    return canonical_text(name[:-2])


def build_vocabulary(labels: Iterable[str]) -> list[str]:
    return [BLANK, *sorted(set("".join(canonical_text(x) for x in labels)))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_splits(seed: int) -> dict[str, str]:
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


def _label_splits(labels_by_group: dict[str, set[str]], seed: int) -> dict[str, str]:
    """Create an 80/10/10 lexical split within each phrase-length group."""
    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    for group in sorted(labels_by_group):
        labels = sorted(labels_by_group[group])
        rng.shuffle(labels)
        n_eval = len(labels) // 10
        n_train = len(labels) - 2 * n_eval
        for label in labels[:n_train]:
            mapping[label] = "train"
        for label in labels[n_train : n_train + n_eval]:
            mapping[label] = "val"
        for label in labels[n_train + n_eval :]:
            mapping[label] = "test"
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
    labels = sorted({record["label"] for record in records})
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


def create_label_disjoint_manifest(
    data_root: Path, output_path: Path, seed: int = 20260908
) -> dict:
    """Split complete labels so validation and test phrases are unseen in training.

    Labels are stratified by the dataset's 1/2/3/n-gram directories. Exact file
    duplicates are removed before assignment. Evaluation labels may be new, but
    every character used by them must occur in at least one training label.
    """
    paths = sorted(data_root.rglob("*.csv"))
    seen_hashes: dict[str, str] = {}
    unique: list[tuple[Path, str, str]] = []
    duplicates = []
    labels_by_group: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        digest = _sha256(path)
        relative = path.relative_to(data_root).as_posix()
        if digest in seen_hashes:
            duplicates.append({"path": relative, "duplicate_of": seen_hashes[digest]})
            continue
        seen_hashes[digest] = relative
        label = label_from_path(path)
        group = relative.split("/", 1)[0]
        labels_by_group[group].add(label)
        unique.append((path, digest, label))

    label_split = _label_splits(labels_by_group, seed)
    records = []
    for path, digest, label in unique:
        relative = path.relative_to(data_root).as_posix()
        records.append(
            {
                "path": relative,
                "label": label,
                "source_id": path.stem,
                "split": label_split[label],
                "sha256": digest,
            }
        )

    split_labels = {
        split: sorted(label for label, assigned in label_split.items() if assigned == split)
        for split in ("train", "val", "test")
    }
    train_characters = set("".join(split_labels["train"]))
    val_characters = set("".join(split_labels["val"]))
    test_characters = set("".join(split_labels["test"]))
    evaluation_characters = val_characters | test_characters
    missing_val_characters = sorted(val_characters - train_characters)
    missing_test_characters = sorted(test_characters - train_characters)
    missing_characters = sorted(evaluation_characters - train_characters)
    if missing_characters:
        raise ValueError(
            "Label-disjoint split contains evaluation characters absent from training: "
            + repr(missing_characters)
        )

    vocabulary = build_vocabulary(split_labels["train"])
    counts = defaultdict(int)
    for record in records:
        counts[record["split"]] += 1
    payload = {
        "schema_version": 1,
        "seed": seed,
        "data_root": data_root.name,
        "split_unit": "complete canonical label stratified by phrase-length directory",
        "evaluation_protocol": "lexical-disjoint open-phrase recognition",
        "writer_identity_confirmed": False,
        "records": records,
        "duplicates_removed": duplicates,
        "vocabulary": vocabulary,
        "labels_by_split": split_labels,
        "statistics": {
            "labels": len(label_split),
            "train_labels": len(split_labels["train"]),
            "val_labels": len(split_labels["val"]),
            "test_labels": len(split_labels["test"]),
            "label_overlap_train_val": len(set(split_labels["train"]) & set(split_labels["val"])),
            "label_overlap_train_test": len(set(split_labels["train"]) & set(split_labels["test"])),
            "vocabulary_including_blank": len(vocabulary),
            "train_character_count": len(train_characters),
            "val_character_count": len(val_characters),
            "test_character_count": len(test_characters),
            "val_characters_missing_from_train": missing_val_characters,
            "test_characters_missing_from_train": missing_test_characters,
            "evaluation_characters_missing_from_train": missing_characters,
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "exact_duplicates_removed": len(duplicates),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
