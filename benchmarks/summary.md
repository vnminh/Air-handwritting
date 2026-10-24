| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| lexical | alam_cnn | xy | 1,192,152 | 3 | 84.04 ± 1.36 | 100.64 ± 0.57 | 0.00 ± 0.00 |
| lexical | alam_lstm | xy | 277,328 | 3 | 82.26 ± 0.90 | 100.49 ± 1.34 | 0.00 ± 0.00 |
| lexical | carbune_bilstm | delta | 443,480 | 3 | 61.50 ± 12.49 | 97.10 ± 4.01 | 1.45 ± 1.27 |
| lexical | ott_cnn_bilstm | xy | 225,968 | 3 | 45.89 ± 4.61 | 87.82 ± 3.45 | 11.10 ± 0.39 |
| lexical | ours | dynamics_fourier | 1,580,392 | 3 | 20.56 ± 0.94 | 55.62 ± 1.43 | 33.23 ± 3.20 |
| lexical | spec_dynf | dyn+fourier | 1,580,392 | 1 | 21.14 | 55.76 | 34.72 |
| source | alam_cnn | xy | 1,209,180 | 3 | 9.19 ± 5.26 | 11.52 ± 5.74 | 89.78 ± 3.59 |
| source | alam_lstm | xy | 311,252 | 3 | 19.34 ± 2.40 | 24.23 ± 3.08 | 78.90 ± 2.97 |
| source | carbune_bilstm | delta | 443,480 | 3 | 41.18 ± 45.18 | 56.94 ± 45.60 | 44.11 ± 46.14 |
| source | ott_cnn_bilstm | xy | 225,968 | 3 | 7.12 ± 1.95 | 16.58 ± 2.45 | 88.52 ± 0.53 |
| source | ours | all | 1,548,267 | 3 | 1.98 ± 0.16 | 5.94 ± 0.53 | 94.48 ± 0.35 |
| source | spec_dynf | dyn+fourier | 1,579,368 | 1 | 1.76 | 5.49 | 94.42 |
| source | spec_spec | spec | 1,550,952 | 1 | 3.03 | 9.20 | 91.87 |
