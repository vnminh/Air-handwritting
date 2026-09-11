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

