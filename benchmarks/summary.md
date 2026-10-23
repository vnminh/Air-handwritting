| Protocol | Method | Repr. | Params | Seeds | Test CER | Test WER | Exact |
|---|---|---|---:|---:|---:|---:|---:|
| lexical | alam_cnn | xy | 1,192,152 | 1 | 83.32 | 100.45 | 0.00 |
| lexical | alam_lstm | xy | 277,328 | 1 | 81.63 | 99.35 | 0.00 |
| lexical | ours | dynamics_fourier | 1,580,392 | 1 | 21.47 | 57.12 | 29.59 |
| source | alam_cnn | xy | 1,209,180 | 1 | 14.67 | 17.33 | 86.38 |
| source | alam_lstm | xy | 311,252 | 1 | 20.81 | 26.10 | 76.76 |
| source | carbune_bilstm | delta | 443,480 | 1 | 91.14 | 100.00 | 0.00 |
| source | ott_cnn_bilstm | xy | 225,968 | 1 | 5.45 | 14.26 | 88.58 |
| source | ours | all | 1,548,267 | 2 | 2.07 ± 0.07 | 6.24 ± 0.11 | 94.29 ± 0.18 |
