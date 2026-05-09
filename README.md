# When and Why Transcoders Are More Interpretable

Transcoders (TCs) have been reported to outperform Sparse Autoencoders (SAEs) on automated interpretability benchmarks — but the structural reasons for this gap remain poorly understood. This repository contains code and experiments for our paper investigating when and why this advantage holds, using a layer-wise analysis across GPT2-small and Pythia-160M/410M.

## Key Findings
* TC's interpretability advantage is layer- and model-dependent, not a general property
* Decoder subspaces of SAE and TC are nearly identical — the gap is not explained by dictionary geometry
* TC features activate over more geometrically consistent inputs (higher sim<sub>x</sub>), and this consistency correlates significantly with per-feature auto-interp scores
* FFN neighborhood distortion (sim<sub>in</sub> − sim<sub>out</sub>) directionally aligns with both the interpretability gap and the consistency gap across layers