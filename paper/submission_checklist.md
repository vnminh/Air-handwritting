# SOICT submission checklist

- Replace `Anonymous Authors` and `Anonymous Institution` in `main.tex`.
- Generate the final metrics table with `python3 scripts/render_paper_results.py` after downloading `/experiments` from the Modal results volume.
- Add the final architecture, Fourier-scale, preprocessing, length, and robustness tables from the completed JSON artifacts.
- Inspect the prediction JSONL files for representative Vietnamese diacritic errors and include the confusion/error analysis.
- Compile with the SOICT/LLNCS class supplied by the conference and verify the page limit.
- Archive the frozen manifest, source code, environment versions, and Modal app IDs with the submission.
