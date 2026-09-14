# Inference examples: validation-selected Fourier K=2 checkpoint

The examples below were executed end to end on Modal with
`models/best_fourier_k2.pt`, including checkpoint-defined preprocessing and
feature extraction.

| Trajectory | Reference | Prediction | Mean gate weights (spatial/dynamic/Fourier) |
|---|---|---|---|
| `2_grams/bao giờ_w/17.csv` | `bao giờ` | `bao giờ` | 0.181 / 0.409 / 0.411 |
| `n_grams/ngọn núi cao vời vợi_w/3.csv` | `ngọn núi cao vời vợi` | `ngọn núi cao vời vợi` | 0.162 / 0.454 / 0.384 |
| `1_gram/bí_w/21.csv` | `bí` | `bú` | 0.229 / 0.315 / 0.456 |

The final row is a useful failure case: the model preserves the consonant but
confuses the accented vowel. This is representative of the Vietnamese
diacritic errors discussed in the paper.
