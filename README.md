# Vietnamese Air-Writing Recognition

This repository contains the reproducible SOICT experiment pipeline for
character-level Vietnamese air-writing recognition. The experimental claim is
trajectory representation: spatial coordinates, motion dynamics, and
multi-scale Fourier features are fused before a Conformer-CTC encoder.

The frozen manifest is [manifests/split_seed_20260908.json](manifests/split_seed_20260908.json).
It has 18,202/2,276/2,276 train/validation/test samples, 660 labels, globally
disjoint numeric source IDs, and six exact duplicate files removed. The
repository does not contain metadata proving that those IDs are writer IDs, so
the paper must describe the split as source-disjoint rather than writer-
independent.

## Local checks

The Modal image supplies PyTorch, NumPy, and SciPy. The manifest itself can be
regenerated with `python3 -m scripts.make_manifest`.

## Modal workflow

```bash
modal volume create vni-airwriting-data
modal volume create vni-airwriting-results
modal volume put vni-airwriting-data /tmp/vni_airwriting.tar.gz /staging/vni_airwriting.tar.gz
modal run modal_app.py::prepare_data_volume
modal run modal_app.py::prepare_exact_cache
modal run --detach modal_app.py --suite core
modal run --detach modal_app.py --suite extended
modal run --detach modal_app.py::robustness
# To obtain the main paper result first, run the short priority suite:
modal run --detach modal_app.py --suite priority
modal volume get vni-airwriting-results /experiments ./artifacts
```

`core` runs architecture, representation, fusion, positional-encoding, and
augmentation comparisons, including seed 43/44 confirmation runs for the main
baselines and proposed model. `extended` runs Fourier-scale, preprocessing,
and resampling-length studies. `robustness` evaluates the saved Transformer,
Conformer, and proposed checkpoints under rotation, noise, scale, and temporal
warp perturbations. Each run writes a checkpoint, JSON metrics, predictions,
and training history; completed runs are skipped safely on restart.

The first completed Fourier-scale comparison selects K=2 on validation. Its
seed-42 checkpoint currently gives test CER 1.77%, WER 5.55%, and exact
sequence accuracy 94.77%; the aggregate table will be refreshed after the
remaining detached jobs finish.

Current detached runs: core `ap-I9nRQY6b3HE0seftt5DWHs`, extended
`ap-KbHzIj9AqusCgXl7YsStmE`, priority `ap-6IX6i94lLL7zFIOZ0u3kGY`, and
robustness `ap-cGQ9dQDsyXCqFschif3CVU`. Inspect them with `modal app logs
<app-id> --tail 100 --timestamps`.
