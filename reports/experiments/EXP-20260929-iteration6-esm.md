# Sixth iteration: frozen ESM label-wise classifiers

Date: 2026-09-29. These are local validation results, not competition scores.
The fixed `iteration4_tail` validation set has 1,062 proteins and 500 labels.
Threshold policies are fitted on one half and evaluated on the other for each
of seeds 17, 31, 42, 73, and 101. Production eligibility requires continuous
Macro AUC >= 0.8280390455031759; the requested Macro F1 is strictly > 0.5.

## Reproducible stages

```shell
python -m src.esm_embeddings --config configs/iteration6_esm2_35m_embeddings.json
python -m src.train_esm_labelwise --config configs/iteration6_esm2_labelwise_screen.json
python -m src.train_esm_labelwise --config configs/iteration6_esm2_labelwise_full.json
python -m src.sweep_esm_fusion --config configs/iteration6_esm_fusion_core.json
python -m src.sweep_homology --config configs/iteration6_rebuild_homology_tail.json
python -m src.sweep_shift --config configs/iteration6_rebuild_tail_sgd.json
python -m src.sweep_homology_blend --config configs/iteration6_rebuild_homology_sgd_blend.json
python -m src.sweep_esm_fusion --config configs/iteration6_esm_fusion_rebuilt.json
```

The embedding run used `facebook/esm2_t12_35M_UR50D` at revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`, model SHA-256
`e35647818e0e064351d4531ed480d225a002567b4b2b93ad3a9246d753150fc0`,
CUDA AMP, 1,022-residue windows with 128 overlap, and 960-dimensional
mean/max representations. The local GPU is an RTX 4060 Laptop GPU (8 GB),
with PyTorch 2.9.1+cu126 and Transformers 4.57.6. The training manifest
records 113,796 proteins; the downstream label-wise run produces 1,062 x 500
validation and 28,450 x 500 test scores. The full label-wise fit took
3,208 seconds. Historical SGD reconstruction uses CPU-only SGDClassifier.

## Results

| Run / candidate | Five-seed Macro F1 | Continuous Macro AUC | Mean positive rate | AUC eligible |
| --- | ---: | ---: | ---: | --- |
| 084 AUC rollback | 0.317015 | 0.828653 | 0.097425 | yes |
| 084 ESM alone | 0.144834 | 0.703907 | 0.245893 | no |
| 086 homology alone | 0.334910 | 0.799506 | 0.090322 | no |
| 087 rebuilt SGD (legacy crossfit measure) | 0.309655 | 0.806738 | 0.175957 | no |
| 088 homology + 10% SGD | 0.335063 | 0.821086 | 0.091857 | no |
| 089 AUC rollback + homology rollback | 0.323623 | 0.839261 | 0.103321 | yes |
| **089 selected adaptive** | **0.324243** | **0.828124** | **0.096170** | **yes** |

The 087 rebuild differs slightly from the earlier 067 result (F1 0.309670,
AUC 0.806789); it is not bit-for-bit identical. Run 088's highest-F1 grid
point is 0.335199, but its tie-tolerance rule selected the above higher-AUC
point. For run 089, the selected five-seed F1 standard deviation is 0.013719
and the minimum is 0.301616. All aggregate results and per-seed diagnostics
are available locally under the corresponding `artifacts/metrics/` prefixes.

The selected adaptive policy uses only homology rollback and its pairwise
blend with AUC rollback; none of its 5,000 fitted label-fold policies selects
ESM. Frozen ESM therefore did not pass the usefulness gate (F1 > 0.335178
without sacrificing AUC). The best AUC-eligible F1 is **0.324243**, not > 0.5.
Its AUC is 0.000085 above the minimum but below the standalone AUC rollback
(0.828653); there is no stable joint improvement claim. The 0.839261-AUC
pairwise rollback offers a higher-AUC alternative at F1 0.323623. No new
competition submission was generated. Confirm the competition's permission
for external pretrained ESM weights before using them for a submission.

Next iteration should first examine label-support diagnostics and independent
validation of candidate selection; a per-label fusion weight search remains
unimplemented and requires a separate bounded design and evaluation. Given
the weak frozen ESM scores, GPU fine-tuning is not justified by this run alone.
