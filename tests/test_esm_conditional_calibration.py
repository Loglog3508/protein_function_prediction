import numpy as np
import pandas as pd
import pytest
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.esm_conditional_calibration import (
    fit_variant_thresholds,
    select_support_prior,
    run_conditional_calibration as run_in_current_process,
)


def run_conditional_calibration(config):
    """Exercise the runner in a fresh CPU process, as production requires."""
    script = """
import json
import sys
from src.esm_conditional_calibration import run_conditional_calibration
try:
    result = run_conditional_calibration(json.load(sys.stdin))
except (ValueError, FileExistsError) as error:
    print(json.dumps({'error_type': type(error).__name__, 'message': str(error)}))
    raise SystemExit(1)
print(json.dumps({'summary_path': str(result)}))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
        timeout=60,
    )
    assert completed.stdout.strip(), completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    if completed.returncode:
        errors = {"ValueError": ValueError, "FileExistsError": FileExistsError}
        assert result.get("error_type") in errors, completed.stderr
        raise errors[result["error_type"]](result["message"])
    return Path(result["summary_path"])


def test_support_prior_tie_break_prefers_candidate_closest_to_half():
    target = np.array([[1], [0], [1], [0]], dtype=np.uint8)
    scores = np.array([[0.2], [0.2], [0.8], [0.8]], dtype=np.float32)
    prior, diagnostics = select_support_prior(
        target,
        scores,
        np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        label_indices=np.array([0]),
    )
    assert np.isclose(prior, 0.4)
    selected = diagnostics.loc[np.isclose(diagnostics["threshold"], 0.4), "macro_f1"]
    assert selected.iat[0] == 0.5


def test_middle_variant_reuses_baseline_thresholds_for_other_strata():
    target = np.array(
        [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.uint8
    )
    scores = np.array(
        [[0.8, 0.7, 0.6], [0.3, 0.3, 0.1], [0.1, 0.1, 0.2], [0.8, 0.7, 0.9]],
        dtype=np.float32,
    )
    strata = np.array(["low", "middle", "high"])
    baseline, middle, records = fit_variant_thresholds(
        target,
        scores,
        strata,
        variant="middle_only_stratum_prior",
        prior_grid=np.array([0.2, 0.5, 0.8], dtype=np.float32),
        shrinkage=25.0,
    )
    assert np.array_equal(baseline[0], middle[0])
    assert np.array_equal(baseline[2], middle[2])
    assert not np.array_equal(baseline[1], middle[1])
    assert set(records["stratum"]) == {"low", "middle", "high"}


@pytest.fixture
def calibration_case(tmp_path):
    labels = ["label_0", "label_1", "label_2"]
    ids = np.array(["P1", "P2", "P3", "P4", "P5", "P6"])
    target = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=np.uint8,
    )
    score_path = tmp_path / "scores.npz"
    np.savez(
        score_path,
        validation_scores=np.array(
            [[0.9, 0.1, 0.2], [0.95, 0.8, 0.3], [0.7, 0.2, 0.8], [0.1, 0.7, 0.4], [0.3, 0.2, 0.9], [0.8, 0.3, 0.2]],
            dtype=np.float32,
        ),
        validation_ids=ids,
        label_columns=np.array(labels),
        epoch=np.array(6),
    )
    train = pd.DataFrame(
        {
            "protein_id": ids,
            "label_0": target[:, 0],
            "label_1": target[:, 1],
            "label_2": target[:, 2],
        }
    )
    training = train.copy()
    training['protein_id'] = [f'T{i}' for i in range(6)]
    train = pd.concat([train, training], ignore_index=True)
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    training_ids_path = tmp_path / "training_ids.csv"
    pd.DataFrame({"protein_id": training['protein_id']}).to_csv(training_ids_path, index=False)
    validation_ids_path = tmp_path / 'validation_ids.csv'
    pd.DataFrame({'protein_id': ids}).to_csv(validation_ids_path, index=False)
    output = tmp_path / "metrics" / "EXP-136"

    config={
            "execution": {"enabled": True, "cpu_only": True},
            "input": {
                "scores_path": str(score_path),
                "scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "epoch": 6,
                "train_path": str(train_path),
                "training_ids": str(training_ids_path),
                "validation_ids": str(validation_ids_path),
                "expected_validation_rows": 6,
                "expected_label_count": 3,
            },
            "output_prefix": str(output),
            "output_dir": str(tmp_path / 'run'),
            "calibration": {
                "seeds": [17, 31, 42, 73, 101],
                "shrinkage": 25.0,
                "variants": [
                    {"name": "baseline_global_prior"},
                    {"name": "middle_only_stratum_prior"},
                    {"name": "all_strata_prior"},
                ],
            },
        }
    return config, tmp_path


def test_run_conditional_calibration_writes_audited_outputs(calibration_case):
    config, tmp_path = calibration_case
    summary_path = run_conditional_calibration(config)
    assert summary_path.exists()
    summary = __import__("json").loads(summary_path.read_text())
    assert summary["raw_metrics_equal_across_variants"] is True
    assert {"baseline_global_prior", "middle_only_stratum_prior", "all_strata_prior"} == set(summary["variants"])
    assert (tmp_path / "metrics" / "EXP-136-strata.csv").exists()
    assert (tmp_path / "metrics" / "EXP-136-thresholds.csv").exists()
    assert summary['baseline_matches_historical_predictions'] is True
    assert summary['runtime']['torch_imported'] is False
    strata = pd.read_csv(tmp_path / 'metrics' / 'EXP-136-strata.csv')
    for _, rows in strata.groupby('stratum'):
        assert rows.raw_AUC.nunique() == rows.raw_AP.nunique() == 1
    assert (tmp_path / 'metrics' / 'EXP-136-prior-grid.csv').exists()
    assert (tmp_path / 'run' / 'evaluation-predictions.npz').exists()


def test_count_aggregation_uses_mean_label_counts_but_f1_uses_mean_seed_f1():
    from src.esm_conditional_calibration import aggregate_label_results
    # Truth support = 1 for each label. Mean predictions = [1.5, 0.5].
    # Seed-level overprediction/excess averages would incorrectly give 0.5/1.
    frame = pd.DataFrame([
        {'variant': 'baseline_global_prior', 'seed': 17, 'label': 'a', 'training_tertile': 'middle', 'true_support': 1, 'predicted_positive': 3, 'tp': 1, 'fp': 2, 'fn': 0, 'f1': .5},
        {'variant': 'baseline_global_prior', 'seed': 17, 'label': 'b', 'training_tertile': 'low', 'true_support': 1, 'predicted_positive': 0, 'tp': 0, 'fp': 0, 'fn': 1, 'f1': 0},
        {'variant': 'baseline_global_prior', 'seed': 31, 'label': 'a', 'training_tertile': 'middle', 'true_support': 1, 'predicted_positive': 0, 'tp': 0, 'fp': 0, 'fn': 1, 'f1': 0},
        {'variant': 'baseline_global_prior', 'seed': 31, 'label': 'b', 'training_tertile': 'low', 'true_support': 1, 'predicted_positive': 1, 'tp': 1, 'fp': 0, 'fn': 0, 'f1': 1},
    ])
    strata, labels = aggregate_label_results(frame, validation_rows=4)
    all_labels = strata.loc[strata.stratum == 'all'].iloc[0]
    assert all_labels.overpred_labels == 1
    assert all_labels.excess == .5
    assert all_labels.calF1 == .375
    assert all_labels.FP == 1
    assert all_labels.FN == 1
    assert all_labels.cal_pos_rate == .25
    assert labels.loc[labels.label == 'a', 'predicted_positive'].item() == 1.5


def test_disabled_config_cannot_bypass_cli_via_runner(calibration_case):
    config, tmp_path = calibration_case
    config['execution']['enabled'] = False
    with pytest.raises(ValueError, match='enabled'):
        run_in_current_process(config)
    assert not list((tmp_path / 'metrics').glob('*'))


def test_runner_rejects_a_process_with_torch_already_imported(calibration_case, monkeypatch):
    config, tmp_path = calibration_case
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(ValueError, match="fresh CPU evaluation process"):
        run_in_current_process(config)
    assert not (tmp_path / "metrics").exists()


@pytest.mark.parametrize('mismatch', ['validation_order', 'label_order', 'split_overlap', 'score_hash', 'epoch'])
def test_mismatched_bound_inputs_fail_before_outputs(calibration_case, mismatch):
    config, tmp_path = calibration_case
    if mismatch == 'validation_order':
        path = config['input']['validation_ids']
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    elif mismatch == 'split_overlap':
        pd.DataFrame({'protein_id': ['P1']}).to_csv(config['input']['training_ids'], index=False)
    elif mismatch == 'score_hash':
        config['input']['scores_sha256'] = '0' * 64
    else:
        path = config['input']['scores_path']
        with np.load(path, allow_pickle=False) as saved:
            payload = {k: saved[k] for k in saved.files}
        if mismatch == 'label_order':
            payload['label_columns'] = payload['label_columns'][::-1]
        else:
            payload['epoch'] = np.array(7)
        np.savez(path, **payload)
        config['input']['scores_sha256'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    with pytest.raises(ValueError):
        run_conditional_calibration(config)
    assert not (tmp_path / 'metrics').exists()


def test_binary_validation_happens_before_uint8_conversion():
    with pytest.raises(ValueError, match='binary'):
        fit_variant_thresholds(np.array([[1.5], [0]]), np.array([[.8], [.2]]), np.array(['middle']), variant='middle_only_stratum_prior')


def test_baseline_and_unchanged_strata_reproduce_historical_thresholds():
    from src.rank_fusion import fit_rank_fusion_thresholds
    target = np.array([[1,0,1], [0,1,0], [0,0,1], [1,1,0]], dtype=np.uint8)
    scores = np.array([[.7,.8,.3], [.2,.6,.1], [.4,.3,.8], [.8,.2,.7]], dtype=np.float32)
    expected, _, _ = fit_rank_fusion_thresholds(target, scores, shrinkage=25)
    baseline, middle, _ = fit_variant_thresholds(target, scores, np.array(['low','middle','high']), variant='middle_only_stratum_prior')
    np.testing.assert_array_equal(baseline, expected)
    np.testing.assert_array_equal(middle[[0,2]], expected[[0,2]])


def test_zero_positive_label_and_degenerate_stratum_record_prior_fallback():
    target = np.array([[1,0,1], [0,0,0]], dtype=np.uint8)
    scores = np.array([[.8,.9,.7], [.2,.6,.1]], dtype=np.float32)
    baseline, conditional, records = fit_variant_thresholds(target, scores, np.array(['low','middle','high']), variant='all_strata_prior')
    row = records.loc[records.label_index == 1].iloc[0]
    assert row.prior_fallback
    assert row.zero_positive_fallback
    assert conditional[1] == pytest.approx(row.global_prior)


def test_prior_selection_uses_historical_float32_tie_rule():
    from src.rank_fusion import fit_rank_fusion_thresholds
    target = np.array([[1], [0], [1], [0]], dtype=np.uint8)
    scores = np.full((4,1), .99, dtype=np.float32)
    candidates = np.arange(.02, 1, .02, dtype=np.float32)
    _, historical, _ = fit_rank_fusion_thresholds(target, scores, global_candidates=candidates)
    selected, _ = select_support_prior(target, scores, candidates, label_indices=np.array([0]))
    assert selected == historical


def test_existing_outputs_are_never_overwritten(calibration_case):
    config, tmp_path = calibration_case
    directory = tmp_path / 'metrics'
    directory.mkdir()
    old = directory / 'EXP-136-per-label.csv'
    old.write_text('keep me', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run_conditional_calibration(config)
    assert old.read_text(encoding='utf-8') == 'keep me'


def test_degenerate_evaluation_auc_is_explicitly_missing():
    from src.esm_conditional_calibration import _raw_metrics
    auc, ap = _raw_metrics(np.array([[0],[0]], dtype=np.uint8), np.array([[.2],[.3]], dtype=np.float32), np.array([0]))
    assert np.isnan(auc) and np.isnan(ap)


def test_evaluation_truth_does_not_change_selection_fold_thresholds(calibration_case):
    from src.diagnose_esm_epochs import fold_indices
    config, tmp_path = calibration_case
    first = run_conditional_calibration(config)
    original = pd.read_csv(tmp_path / 'metrics' / 'EXP-136-thresholds.csv')
    _, evaluation = next(iter(fold_indices(6, 17)))
    data = pd.read_csv(config['input']['train_path'])
    row = int(evaluation[0])
    data.loc[row, 'label_0'] = 1 - data.loc[row, 'label_0']
    data.to_csv(config['input']['train_path'], index=False)
    config['output_prefix'] = str(tmp_path / 'metrics' / 'heldout-perturbed')
    config['output_dir'] = str(tmp_path / 'perturbed-run')
    run_conditional_calibration(config)
    changed = pd.read_csv(tmp_path / 'metrics' / 'heldout-perturbed-thresholds.csv')
    original = original.loc[(original.seed == 17) & (original.fold == 0)].reset_index(drop=True)
    changed = changed.loc[(changed.seed == 17) & (changed.fold == 0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, changed)


def test_prior_grid_and_saved_predictions_allow_independent_reconstruction(calibration_case):
    from sklearn.metrics import roc_auc_score, average_precision_score
    config, tmp_path = calibration_case
    run_conditional_calibration(config)
    with np.load(config['input']['scores_path']) as saved:
        scores=saved['validation_scores'];ids=saved['validation_ids'];labels=saved['label_columns']
    truth=pd.read_csv(config['input']['train_path']).set_index('protein_id').loc[ids, labels].to_numpy()
    with np.load(tmp_path/'run'/'evaluation-predictions.npz') as saved:
        predictions={name:saved[name] for name in ('baseline_global_prior','middle_only_stratum_prior','all_strata_prior')}
    strata=pd.read_csv(tmp_path/'metrics'/'EXP-136-strata.csv')
    for name,predicted in predictions.items():
        row=strata.loc[(strata.variant==name)&(strata.stratum=='all')].iloc[0]
        mean_counts=predicted.sum(axis=1).mean(axis=0)
        support=truth.sum(axis=0)
        assert row.overpred_labels==int((mean_counts>support).sum())
        assert row.excess==pytest.approx(np.maximum(mean_counts-support,0).sum())
        assert row.raw_AUC==pytest.approx(np.mean([roc_auc_score(truth[:,i],scores[:,i]) for i in range(3)]))
        assert row.raw_AP==pytest.approx(np.mean([average_precision_score(truth[:,i],scores[:,i]) for i in range(3)]))
        tp=(predicted*truth).sum(axis=1).astype(float)
        denom=predicted.sum(axis=1)+support
        f1=np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
        assert row.calF1==pytest.approx(f1.mean())
