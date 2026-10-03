"""Paired protocols for five cap interventions; no training or prediction on import."""
from __future__ import annotations

import copy

SUITE = [
    {'id': 'EXP-20261002-131-esm35-asl-cap5-epoch6', 'config': 'configs/iteration20_esm35_asl_cap5_epoch6.json', 'baseline_config': 'configs/iteration17_esm35_last4_full_epoch10.json', 'baseline_number': 127, 'epochs': 6, 'cap': 5.0},
    {'id': 'EXP-20261002-132-esm35-asl-cap2-epoch6', 'config': 'configs/iteration20_esm35_asl_cap2_epoch6.json', 'baseline_config': 'configs/iteration17_esm35_last4_full_epoch10.json', 'baseline_number': 127, 'epochs': 6, 'cap': 2.0},
    {'id': 'EXP-20261002-133-esm35-asl-balanced-epoch6', 'config': 'configs/iteration20_esm35_asl_balanced_epoch6.json', 'baseline_config': 'configs/iteration17_esm35_last4_full_epoch10.json', 'baseline_number': 127, 'epochs': 6, 'cap': 'balanced', 'weight_definition': 'max(1, negatives/positives); no upper limit; zero positives -> 1'},
    {'id': 'EXP-20261002-134-esm35-bce-cap5-epoch6', 'config': 'configs/iteration21_esm35_bce_cap5_epoch6.json', 'baseline_config': 'configs/iteration19_esm35_bce_full_epoch6.json', 'baseline_number': 130, 'epochs': 6, 'cap': 5.0},
    {'id': 'EXP-20261002-135-esm150-asl-cap5-epoch8', 'config': 'configs/iteration21_esm150_asl_cap5_epoch8.json', 'baseline_config': 'configs/iteration18_esm150_last4_full_epoch8.json', 'baseline_number': 129, 'epochs': 8, 'cap': 5.0},
]

BASELINE_IDS = {127: 'EXP-20260930-127-esm35-last4-full-epoch10',
                130: 'EXP-20261001-130-esm35-bce-last4-full-epoch6',
                129: 'EXP-20261001-129-esm150-last4-full-epoch8'}


def paired_contract(task: dict, baseline: dict, candidate: dict) -> dict:
    """Reject wrong pairings or any unapproved training/model protocol change."""
    if baseline['experiment_id'] != BASELINE_IDS[task['baseline_number']]:
        raise ValueError('wrong paired baseline identity')
    if candidate['experiment_id'] != task['id']:
        raise ValueError('wrong candidate identity')
    if candidate['output_dir'] != 'artifacts/runs/' + task['id']:
        raise ValueError('candidate needs its own output directory')
    expected = copy.deepcopy(baseline)
    expected.update(experiment_id=task['id'], output_dir='artifacts/runs/' + task['id'])
    expected['training'].update(epochs=task['epochs'], positive_weight_cap=task['cap'])
    if candidate != expected:
        raise ValueError('single-variable protocol changed beyond cap, identity and approved epoch budget')
    if baseline['training']['positive_weight_cap'] != 20 or not candidate['training'].get('save_epoch_scores'):
        raise ValueError('baseline cap20 and every-epoch scores are required')
    if task['epochs'] > baseline['training']['epochs']:
        raise ValueError('paired baseline does not cover the full common epoch budget')
    return {'paired_baseline_id': baseline['experiment_id'], 'candidate_id': candidate['experiment_id'],
            'epochs': task['epochs'], 'changed_training_fields': ['positive_weight_cap'],
            'budget_note': 'first six baseline epochs reused' if task['baseline_number'] == 127 else 'same full epoch budget'}


def support_groups(support):
    """EXP-128 stable support tertiles; tie order is the original label order."""
    import numpy as np
    values = np.asarray(support)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('training support must be a finite nonnegative vector')
    groups = np.empty(len(values), dtype='<U6')
    for name, indices in zip(['low', 'middle', 'high'], np.array_split(np.argsort(values, kind='stable'), 3)):
        groups[indices] = name
    return groups
