# Content decisions for `Paper_XLA.docx`

## Retained and rewritten

- Motivation for touchless air-writing and the sensor/radar/camera taxonomy.
- Lack of tactile feedback, epenthetic strokes, and spatiotemporal variation.
- Classical, image-based, sequence-based, Transformer, Conformer, CTC, and
  Fourier-feature related work.
- Dataset motivation, organization, coordinate-sequence format, and the
  difficulty of Vietnamese diacritics.
- Rotation, stretching, Gaussian-noise, and temporal-warp augmentation.
- Savitzky--Golay filtering, spline interpolation, normalization, and
  arc-length resampling.
- CER definition, qualitative error discussion, limitations, and future work.
- All 18 references, with the original Conformer/CTC numbering error removed.

## Changed to match the reproducible implementation

- 15,500 one-word training samples became all 22,754 non-duplicate samples.
- Directory-based train/test assignment became a global source-ID split of
  18,202/2,276/2,276 trajectories.
- Validation/test augmentation was removed.
- Five-fold offline augmentation was replaced by one deterministic,
  training-only transformed trajectory per run.
- Bounding-box `[0,1]` normalization became centroid and maximum-radius
  normalization.
- Smoothing spline with `s=2N` became `CubicSpline` interpolation.
- Stretching `[0.8,1.2]` became independent scaling `[0.85,1.15]`.
- Gaussian RFF (`m=10`, `sigma^2=10`) became deterministic dyadic Fourier
  features, with validation selecting `K=2`.
- Eight hand-engineered features became spatial, seven-dimensional dynamic,
  and Fourier branches with adaptive gated fusion.
- Six Conformer layers became four layers with learned clipped relative bias.
- Batch size 128/50 epochs became batch size 512/40 epochs with four warm-up
  epochs and cosine decay.
- The old 15.02% CER table became the source-disjoint architecture and
  ablation results produced by Modal.

## Excluded

- Dataset screenshot containing author/project metadata.
- Old normalization figures whose coordinates are scaled to `[0,1]`.
- Old architecture diagram, training curves, and 15.02% result table.
- Qualitative prediction images produced by the earlier checkpoint.

The augmentation and architecture figures were regenerated from the current
pipeline. Qualitative examples now come from the downloaded Fourier-K2 test
predictions and verified checkpoint inference.
