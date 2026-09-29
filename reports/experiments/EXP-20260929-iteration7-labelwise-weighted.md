# Seventh iteration: constrained label-wise weight search

Date: 2026-09-29. The four original OOF score matrices were restored from the
remote Git LFS branch and the searches were completed locally. These are local
cross-fitted validation results, not competition scores.

## Objective and method

The goal is a five-seed cross-fitted Macro F1 above 0.5 on the fixed
`iteration4_tail` validation cohort, subject to continuous Macro AUC at least
0.8280390455031759. Each selection half searches the four original score
sources, plus all two-source 25/75, 50/50 and 75/25 mixtures, with a separate
threshold for each label with at least 20 selection-half positives. Labels
below that support threshold share a source and threshold within their support
stratum. The opposite half is used only for evaluation, and the roles are
reversed for seeds 17, 31, 42, 73 and 101. Existing rollback candidates remain
in the leaderboard. No ESM retraining or platform submission is included.

The initial run used `configs/iteration7_labelwise_weighted.json`. A second
run added the per-label AUC gate and used
`configs/iteration7_labelwise_weighted_auc_gated.json`; a third run tested a
0.005 per-label AUC tolerance. All runs use new experiment IDs and refuse
output collisions.

## Results

| Run / candidate | Five-seed Macro F1 | Continuous Macro AUC | AUC eligible | Status |
| --- | ---: | ---: | --- | --- |
| 089 selected adaptive | 0.324243 | 0.828124 | yes | Sixth-iteration reference |
| 090 labelwise weighted | 0.323268 | 0.825298 | no | F1-only label search |
| 093 labelwise weighted, AUC gate | 0.321016 | 0.830271 | yes | AUC preserved, F1 decreased |
| 094 labelwise weighted, AUC tolerance 0.005 | 0.320925 | 0.830225 | yes | No F1 gain |
| 095 labelwise weighted, threshold shrinkage 10 | 0.323465 | 0.826109 | no | Shrinkage did not recover fifth-iteration F1 |

The best measured candidate remains the fifth-iteration homology + 10% SGD
blend (`EXP-20260928-068`): F1 `0.335178`, continuous AUC `0.821132`. The
existing platform candidate is
`artifacts/submissions/EXP-20260928-070-iteration5-final-homology-sgd/submit_template_v1.csv`.

The four inputs are now present as Git LFS working-tree files. Their matrices
are each 1062 x 500, match the canonical validation ID order, and contain
finite values in [0, 1].

## Next iteration target

The seventh iteration did not improve the best local F1 and did not approach
0.5. For a single platform submission, retain the fifth-iteration candidate
until a new model produces a higher cross-fitted F1. Any ESM-based submission
still requires confirmation of the competition's external-pretrained-weight
rules.
