# Preserved checkpoints

## `best_validation_selected.pt`

Primary source-disjoint checkpoint selected by the lowest validation CER among
all completed seed-42 configurations. It uses the three gated branches,
relative position bias, Fourier scale K=4, and no training augmentation.

- Best epoch selected by validation CER: 39
- Validation CER: 0.0151011006
- Test CER: 0.0156983193
- Test WER: 0.0496780129
- Exact sequence accuracy: 0.9463971880
- SHA-256: `08b1ebc27d122af8701a0ae7f354b4ecf387e2db0714a273e025f5db730662a9`

## `best_fourier_k2.pt`

Validation-selected checkpoint within the augmented Fourier-scale comparison.
It contains the complete PyTorch state dictionary, model configuration, and
88 output tokens (86 corpus characters, space, and CTC blank).

- Validation CER: 0.0185137787
- Test CER: 0.0176606092
- Test WER: 0.0555044465
- Exact sequence accuracy: 0.9477152900
- SHA-256: `fe460023f0037dfe58697c3f795cc61dca040b21dda02297e200274e8bc46e4b`

## `best_preprocess_spline.pt`

Secondary ablation checkpoint. It has the lowest observed test CER, but it was
not chosen as the primary model because its validation CER is worse than the
Fourier K=2 checkpoint.

- Validation CER: 0.0191963143
- Test CER: 0.0172340244
- Test WER: 0.0561177553
- Exact sequence accuracy: 0.9485940246
- SHA-256: `a8be0d1c3adf08c8c4e5994476feb100dd0bcd740898de5919912ca6e95906f5`

## Exploratory label-disjoint checkpoints

The following checkpoints were produced during an exploratory stricter split
and are retained only as audit artifacts. They are not used in the paper's
source-disjoint tables or claims.

### `best_label_disjoint_fourier_k2.pt`

The same Fourier K=2 architecture retrained from scratch with complete
validation and test labels excluded from training.

- Best epoch selected by validation CER: 38
- Validation CER: 0.3405647098
- Test CER: 0.3316420664
- Test WER: 0.7721682848
- Exact sequence accuracy: 0.1522935780
- SHA-256: `7f5145dc7e6acff33bd2c8652e56301f757b9f4aafd2d0a05f0309d987e793db`

### `best_label_disjoint_validation_selected.pt`

- Validation CER: 0.2848389501
- Test CER: 0.3199261993
- Test WER: 0.7417475728
- Exact sequence accuracy: 0.1637614679
- SHA-256: `0a1ecf8e61fbefba657b4754615618898cb1729d0daa41041b2d1a00d148749a`

Use `python -m airwriting <trajectory.csv>` to run the Fourier-$K=2$ paper
checkpoint. An alternate checkpoint can be supplied explicitly with
`--checkpoint`; the legacy `python scripts/infer_csv.py <trajectory.csv>`
command remains available.
