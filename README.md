# Vietnamese Air-Writing Recognition

Character-level recognition of Vietnamese air-writing trajectories: a hand
traces letters in free space, a camera tracks the fingertip as a sequence of
`(x, y)` coordinates, and the model reads out the Vietnamese text — diacritics
and all.

The core idea is a **representation-adaptive Conformer**. Instead of feeding
raw coordinates into a sequence model, each trajectory is split into three
complementary views — spatial shape, motion dynamics (speed, direction,
curvature), and multi-scale Fourier detail — which a learned gate recombines
*differently at every point along the stroke*. A relative-position Conformer
encoder and a CTC objective turn the fused sequence into text.

![Selected model architecture](paper/figures/best_model_architecture.png)

## Results

We evaluate under two protocols. **Source-disjoint** holds out entire
recording sessions, which is the standard setup — but still lets every test
*phrase* have been seen, word-for-word, during training. **Lexical-disjoint**
is a harder, self-imposed test: none of the test phrases occur anywhere in
training, so the model can't just recall a phrase — it has to compose
characters from motion it has never seen combined that way before.

| Protocol | Configuration | CER ↓ | WER ↓ | Exact accuracy ↑ |
|---|---|---:|---:|---:|
| Source-disjoint | Full model (spatial + dynamics + Fourier, gated) | **1.77%** | 5.55% | 94.77% |
| Lexical-disjoint | Same full model, no changes | 33.16% | 77.22% | 15.23% |
| Lexical-disjoint | Dynamics + Fourier only, dropout 0.2 | **19.52% ± 1.17%** | 52.4% ± 2.0% | 37.2% ± 3.0% |

The lexical-disjoint number is the interesting one: dropping the spatial
(raw-coordinate) branch and increasing dropout — no architecture change —
cuts the error by more than a third (33.16% → 19.52%, averaged over three
random seeds) on phrases the model has truly never encountered. That's
evidence the model is composing characters from motion, not memorizing 660
training labels.

On the backbone side, the Conformer isn't a marginal choice: it beats a
Transformer with the same features by more than 6× lower CER (1.85% vs.
11.79%), and recurrent baselines (BiLSTM, GRU) are far worse still.

## What we found interesting

**The fusion gate genuinely specializes, and it tracks trajectory geometry.**
Across every point of every test trajectory, the dynamic (motion) branch's
gate weight rises specifically where the pen is turning sharply — corners,
diacritic strokes, direction reversals (Spearman ρ = 0.57 between gate weight
and local curvature). The gate isn't a fixed blend; it shifts weight toward
motion detail exactly where shape alone stops being enough.

**Most residual errors are about *which letter*, not *which tone*.**
Decomposing every error into base-letter vs. diacritic/tone components, base
letters account for 5× more errors than tone marks (40.6% vs. 7.3% of all
errors). Vietnamese tone placement — often assumed to be the hard part — is
already handled well; the harder problem is short, low-amplitude strokes that
distinguish similar-looking vowels (e.g. *bí* → *bú*).

**Geometric noise is tolerated well; localization noise is not.** CER stays
under 3% across ±15° rotation, 0.8–1.2× scale, and moderate time-warping, but
degrades sharply under fingertip-tracking jitter — a much more realistic
threat model for a camera-based system than viewpoint or speed variation.

Full experimental detail, all ablations, and the complete write-up are in
[`paper/Paper_XLA.pdf`](paper/Paper_XLA.pdf).

## Quick inference

Two checkpoints ship with the repo — one for each protocol above:

```bash
# Source-disjoint model (best overall accuracy)
python -m airwriting examples/bao_gio.csv \
  --checkpoint models/best_fourier_k2.pt

# Lexical-disjoint model (best at generalizing to unseen phrases)
python -m airwriting examples/bao_gio.csv \
  --checkpoint models/best_lexical_disjoint_dynf_d20.pt
```

```json
{
  "prediction": "bao giờ",
  "raw_points": 156,
  "model_points": 128
}
```

A few more example trajectories (including a failure case) live in
[`examples/`](examples/). See [`models/README.md`](models/README.md) for the
full checkpoint inventory, and the Python API:

```python
from airwriting.inference import AirWritingRecognizer

recognizer = AirWritingRecognizer("models/best_fourier_k2.pt")
print(recognizer.predict_csv("examples/bao_gio.csv")["prediction"])
```

## Installation

```bash
git clone https://github.com/vnminh/Air-handwritting.git
cd Air-handwritting
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

## Repository layout

```text
airwriting/     Data, model, training, and inference code
examples/       Example trajectories for quick inference
manifests/      Frozen train/val/test split definitions
models/         Preserved checkpoints
paper/          The paper (LaTeX + PDF) and its figures
scripts/        Manifest, figure, and inference utilities
modal_app.py    Cloud training/evaluation pipeline (Modal)
```

## Reproducing training

Training runs on [Modal](https://modal.com/) GPUs; `modal_app.py` defines
every experiment suite referenced in the paper (backbone/representation/
fusion ablations, robustness sweeps, and the lexical-disjoint study with its
validation-then-test protocol). Run `modal run modal_app.py --help` after
`modal setup`, or see the suite definitions directly in `modal_app.py`.

## Citation

```bibtex
@inproceedings{vietnamese_airwriting_conformer,
  title  = {Multi-Representation Trajectory Learning for Vietnamese
            Air-Writing Recognition},
  author = {Anonymous Authors},
  year   = {2026}
}
```

## Limitations

The model reads tracked 2D coordinates, not images or video, so it doesn't
account for upstream hand-tracking errors. Long phrases and short
discriminative vowel strokes remain the main failure mode (see the `bí → bú`
example in `examples/`), and the lexical-disjoint test set still draws from
the same 86-character alphabet seen in training — it tests phrase
composition, not zero-shot character learning.
