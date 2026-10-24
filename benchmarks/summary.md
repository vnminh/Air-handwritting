| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| lexical | alam_cnn | xy | 1,192,152 | 1 | 83.32 | 100.45 | 0.00 |
| lexical | alam_lstm | xy | 277,328 | 1 | 81.63 | 99.35 | 0.00 |
| lexical | ott_cnn_bilstm | xy | 225,968 | 1 | 42.65 | 84.47 | 11.06 |
| lexical | ours | dynamics_fourier | 1,580,392 | 2 | 21.04 ± 0.61 | 56.30 ± 1.17 | 32.05 ± 3.47 |
| source | alam_cnn | xy | 1,209,180 | 2 | 11.70 ± 4.21 | 14.36 ± 4.21 | 87.89 ± 2.14 |
| source | alam_lstm | xy | 311,252 | 2 | 20.72 ± 0.13 | 26.01 ± 0.13 | 77.20 ± 0.62 |
| source | carbune_bilstm | delta | 443,480 | 1 | 91.14 | 100.00 | 0.00 |
| source | ott_cnn_bilstm | xy | 225,968 | 1 | 5.45 | 14.26 | 88.58 |
| source | ours | all | 1,548,267 | 2 | 2.07 ± 0.07 | 6.24 ± 0.11 | 94.29 ± 0.18 |
