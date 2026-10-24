| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| lexical | alam_cnn | xy | 1,192,152 | 3 | 84.04 ± 1.36 | 100.64 ± 0.57 | 0.00 ± 0.00 |
| lexical | alam_lstm | xy | 277,328 | 3 | 82.26 ± 0.90 | 100.49 ± 1.34 | 0.00 ± 0.00 |
| lexical | carbune_bilstm | delta | 443,480 | 2 | 67.31 ± 10.46 | 99.38 ± 0.87 | 1.01 ± 1.43 |
| lexical | ott_cnn_bilstm | xy | 225,968 | 2 | 46.91 ± 6.02 | 87.91 ± 4.87 | 10.89 ± 0.23 |
| lexical | ours | dynamics_fourier | 1,580,392 | 3 | 20.56 ± 0.94 | 55.62 ± 1.43 | 33.23 ± 3.20 |
| source | alam_cnn | xy | 1,209,180 | 3 | 9.19 ± 5.26 | 11.52 ± 5.74 | 89.78 ± 3.59 |
| source | alam_lstm | xy | 311,252 | 3 | 19.34 ± 2.40 | 24.23 ± 3.08 | 78.90 ± 2.97 |
| source | carbune_bilstm | delta | 443,480 | 2 | 47.17 ± 62.18 | 54.59 ± 64.23 | 46.02 ± 65.09 |
| source | ott_cnn_bilstm | xy | 225,968 | 3 | 7.12 ± 1.95 | 16.58 ± 2.45 | 88.52 ± 0.53 |
| source | ours | all | 1,548,267 | 3 | 1.98 ± 0.16 | 5.94 ± 0.53 | 94.48 ± 0.35 |
