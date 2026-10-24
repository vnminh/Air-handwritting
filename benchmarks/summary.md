| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| lexical | alam_cnn | xy | 1,192,152 | 2 | 84.47 ± 1.62 | 100.32 ± 0.18 | 0.00 ± 0.00 |
| lexical | alam_lstm | xy | 277,328 | 2 | 82.46 ± 1.17 | 99.75 ± 0.57 | 0.00 ± 0.00 |
| lexical | carbune_bilstm | delta | 443,480 | 1 | 59.91 | 98.77 | 2.02 |
| lexical | ott_cnn_bilstm | xy | 225,968 | 2 | 46.91 ± 6.02 | 87.91 ± 4.87 | 10.89 ± 0.23 |
| lexical | ours | dynamics_fourier | 1,580,392 | 2 | 21.04 ± 0.61 | 56.30 ± 1.17 | 32.05 ± 3.47 |
| source | alam_cnn | xy | 1,209,180 | 3 | 9.19 ± 5.26 | 11.52 ± 5.74 | 89.78 ± 3.59 |
| source | alam_lstm | xy | 311,252 | 3 | 19.34 ± 2.40 | 24.23 ± 3.08 | 78.90 ± 2.97 |
| source | carbune_bilstm | delta | 443,480 | 2 | 47.17 ± 62.18 | 54.59 ± 64.23 | 46.02 ± 65.09 |
| source | ott_cnn_bilstm | xy | 225,968 | 3 | 7.12 ± 1.95 | 16.58 ± 2.45 | 88.52 ± 0.53 |
| source | ours | all | 1,548,267 | 3 | 1.98 ± 0.16 | 5.94 ± 0.53 | 94.48 ± 0.35 |
