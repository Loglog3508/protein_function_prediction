import copy
import json
from pathlib import Path

import pytest

from src.esm_cap_suite import paired_contract, SUITE

ROOT = Path(__file__).resolve().parents[1]


def baseline_config(number=3):
    return json.loads((ROOT / SUITE[number]['baseline_config']).read_text())


def candidate_config(number=3):
    value = baseline_config(number)
    value['experiment_id'] = SUITE[number]['id']
    value['output_dir'] = 'artifacts/runs/' + SUITE[number]['id']
    value['training']['positive_weight_cap'] = 5.0
    return value


@pytest.mark.parametrize('number,expected', [(3, 'EXP-20261001-130-esm35-bce-last4-full-epoch6'), (4, 'EXP-20261001-129-esm150-last4-full-epoch8')])
def test_pair_uses_its_actual_loss_and_model_baseline(number, expected):
    receipt = paired_contract(SUITE[number], baseline_config(number), candidate_config(number))
    assert receipt['paired_baseline_id'] == expected
    assert receipt['changed_training_fields'] == ['positive_weight_cap']


@pytest.mark.parametrize('field,value', [('batch_size', 4), ('gradient_accumulation', 4), ('epochs', 5)])
def test_rejects_silent_protocol_changes(field, value):
    candidate = candidate_config()
    candidate['training'][field] = value
    with pytest.raises(ValueError, match='single-variable'):
        paired_contract(SUITE[3], baseline_config(), candidate)


def test_rejects_cross_baseline_comparison():
    wrong = baseline_config(0)
    with pytest.raises(ValueError, match='baseline'):
        paired_contract(SUITE[3], wrong, candidate_config())


def test_original_asl_budget_may_reuse_first_six_of_ten_epochs():
    base = baseline_config(0)
    candidate = copy.deepcopy(base)
    candidate['experiment_id'] = SUITE[0]['id']
    candidate['output_dir'] = 'artifacts/runs/' + SUITE[0]['id']
    candidate['training'].update(epochs=6, positive_weight_cap=5)
    assert paired_contract(SUITE[0], base, candidate)['epochs'] == 6


def test_refuses_reused_output_directory():
    candidate = candidate_config()
    candidate['output_dir'] = baseline_config()['output_dir']
    with pytest.raises(ValueError, match='output'):
        paired_contract(SUITE[3], baseline_config(), candidate)
