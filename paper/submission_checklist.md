# SOICT submission checklist

- Fill in real author names/affiliations in `paper/Paper_XLA.tex` (currently the placeholders `Authors`/`Institution`) once the double-blind status of the track is confirmed.
- [done] Source-disjoint architecture, representation/fusion, Fourier-scale, preprocessing, length, and robustness tables are filled in from `artifacts/modal/summary.csv` and `artifacts/models/fourier_k2/robustness.json`.
- [done] Gate-specialization analysis (entropy, winning branch, curvature correlation) from `artifacts/gate_analysis.json`.
- [done] Vietnamese orthographic error taxonomy (base-letter vs. diacritic/tone vs. space) from `artifacts/error_taxonomy.json`.
- [done] Lexical-disjoint open-phrase evaluation (Table~8) from `artifacts/lexical_regularization`/the Modal `vni-airwriting-results` volume under `/lexical_regularization`.
- [done] Compiles with `tectonic` (XeTeX-based) using `llncs` + `fontspec`/`Liberation Serif` for full Vietnamese Unicode coverage; 12 pages total, references begin partway through page 11 (within the 12-page body limit). Re-verify page count after any further edits with:
  `tectonic paper/Paper_XLA.tex && python3 -c "from pypdf import PdfReader; print(len(PdfReader('paper/Paper_XLA.pdf').pages))"`
- Re-run with the conference-supplied `llncs.cls`/style files if SOICT distributes a modified version, and re-check the page limit.
- Archive the frozen manifests (`manifests/split_seed_20260908.json`, `manifests/label_disjoint_seed_20260908.json`), source code, environment versions, and Modal app IDs with the submission.
