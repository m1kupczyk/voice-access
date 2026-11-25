| System | Runs | EER (%) | 95% CI of EER (%) | minDCF |
|---|---:|---:|---:|---:|
| GMM-UBM | 5 | 14.81 ± 0.15 | [12.89, 16.48] | 0.9373 ± 0.0041 |
| ECAPA-TDNN | 1 | 0.90 | [0.65, 1.14] | 0.1104 |

VoxCeleb1-O (cleaned), 37611 trials, 40 enrollment speakers. Multi-run systems: mean ± sample std (ddof=1) over runs.
minDCF: normalized, p_target=0.01, c_miss=1.0, c_fa=1.0.
CI: speaker-level percentile bootstrap, 1000 resamples of enrollment speakers, seed 1234; for multi-run systems the CI is of the run-mean EER under shared resamples.
