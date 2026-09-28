# EXP-20260928 Iteration 4 GPU F1/AUC Report

## Objective

Improve five-seed tail cross-fit Macro F1 above 0.5 without reducing the
iteration-3 continuous Macro AUC baseline of `0.8150451732223777`.

## Validation Contract

- Tail rows: 1,062
- Labels: 500 support-stratified labels
- Threshold seeds: 17, 31, 42, 73, 101
- GPU OOF: two multilabel-stratified folds with seed 42
- AUC gate: continuous Macro AUC >= `0.8150451732223777`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU

The fixed seed-42 split under `artifacts/metrics/splits/seed42/` was not
modified.

## Rebuilt Anchor

The iteration-3 score files were absent from this checkout, so the two-fold
tail scores were rebuilt before testing GPU blends.

| Experiment | Model | Continuous AUC | Five-seed Macro F1 |
| --- | --- | ---: | ---: |
| 050 | tail-weight-2 SGD | 0.8067375075 | 0.3096548107 (seed 42) |
| 054 | 10% SGD + 90% KNN | 0.8149906782 | 0.3214841522 +/- 0.0023591999 |

The rebuilt anchor is within `0.0000545` AUC and `0.0000328` F1 of the
iteration-3 recorded values. The original recorded AUC, rather than the
slightly lower rebuilt value, remained the hard eligibility gate.

## GPU OOF Training

Both folds used the wide multiscale CNN, asymmetric BCE, maximum positive
weight 3, and tail-row weight 2. Experiments 052 and 053 selected checkpoints
with their own fold validation labels and are retained only as diagnostic
history. Experiments 058 through 061 fixed training to six epochs, but that
budget was selected after inspecting the earlier outer-fold histories. They are
also diagnostic history rather than an independent confirmation.

| Experiment | Validation rows | Retained epoch | Continuous AUC | F1 at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| 062 fixed-8 fold 0 | 529 | 8 fixed | 0.7605946618 | 0.2131644511 |
| 063 fixed-8 fold 1 | 533 | 8 fixed | 0.7665062259 | 0.2088509144 |

Experiment 064 merged the disjoint fixed-epoch folds and reordered score columns
by label name to match the rebuilt anchor. It verified 1,062 unique IDs, no
missing or extra rows, and 500 unique matching labels.

## Best Eligible Blend

Experiment 065 evaluated the fixed-epoch OOF scores from 0% through 70% GPU
weight, retaining the rollback baseline required by the sweep contract. The best
eligible candidate was:

- 60% rebuilt SGD/KNN anchor + 40% GPU OOF scores
- Threshold shrinkage: 10
- Five-seed Macro F1: `0.32419172996031437`
- Standard deviation: `0.0012013940043014708`
- Minimum seed F1: `0.3225473510169179`
- Continuous Macro AUC: `0.8280390455031759`
- Mean predicted-positive rate: `0.0914015065913371`

Relative to the rebuilt anchor, this adds about `0.00271` Macro F1 and `0.01305`
continuous Macro AUC. It preserves the original AUC gate but does not reach the
0.5 F1 target. This fixed-eight-epoch result is the formal iteration outcome;
the higher fixed-six-epoch diagnostic result is not used for the final claim.

## Bottleneck Diagnosis

Using validation labels to select thresholds in-sample gives an optimistic
Macro F1 ceiling of about `0.3771` for any single uniform blend tested. Allowing
both the GPU weight and threshold to be selected independently per label raises
the optimistic ceiling only to `0.4018`. These are intentionally optimistic,
leaky diagnostics, yet both remain below 0.5. More threshold tuning or adaptive
blending of the existing scores therefore cannot meet the target.

The next technically justified candidate is a frozen pretrained protein
language model representation with a GPU-trained multilabel head. It must stay
local and experimental until external pretrained weights are confirmed as
allowed by the competition or project rules.

## Verification

`python -m pytest -q -p no:cacheprovider` completed with 70 passing tests.
