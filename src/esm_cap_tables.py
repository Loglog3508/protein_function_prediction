"""Compact, provenance-checked handoff tables for the five paired cap controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .esm_cap_suite import SUITE, paired_contract, support_groups
from .diagnose_esm_epochs import prevalence_decomposition

CONTROL = Path('artifacts/runs/EXP-20261002-131-controls')
TABLE1 = Path('reports/experiments/EXP-20261002-cap-attribution-table1-baselines.md')
TABLE2 = Path('reports/experiments/EXP-20261002-cap-attribution-table2-results.md')
GROUPS = {127: 'A', 130: 'B', 129: 'C'}
FIELDS = ['delta_calF1', 'delta_AUC', 'delta_AP', 'delta_cal_pos_rate（pp）',
          'delta_overpred_labels', 'delta_excess', 'delta_FP', 'delta_FN']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def markdown(columns, rows):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def display_cap(task):
    return 'max(1,neg/pos)' if task['cap'] == 'balanced' else str(int(task['cap']))


def summary_path(root, task, epoch):
    suffix = 'asl-control-cap-v1' if task['baseline_number'] == 127 else 'paired-cap-v1'
    return root / 'artifacts/metrics' / f"{task['id']}-epoch{epoch:02d}-{suffix}-summary.json"


def audit_epoch(root, task, epoch):
    """Missing evidence is pending; malformed, changed or wrongly paired evidence fails."""
    config_path = root / task['config']
    baseline_path = root / task['baseline_config']
    config, baseline = read(config_path), read(baseline_path)
    paired_contract(task, baseline, config)
    path = summary_path(root, task, epoch)
    receipt = root / CONTROL / 'verification' / f"{task['id']}-epoch{epoch:02d}-independent-audit.json"
    if not path.exists() or not receipt.exists():
        return None
    result, audit = read(path), read(receipt)
    assert result['epoch'] == audit['epoch'] == epoch
    assert audit['status'] == 'passed' and audit['cpu_only'] and not audit['torch_imported']
    assert result['candidate']['experiment_id'] == audit['candidate_id'] == task['id']
    assert result['baseline']['experiment_id'] == baseline['experiment_id']
    assert audit['source_summary'] == path.relative_to(root).as_posix()
    assert sha(path) == audit['source_summary_sha256']
    assert result['protocol']['seeds'] == audit['seeds_checked'] == [17, 31, 42, 73, 101]
    assert result['protocol']['shrinkage'] == 25 and result['protocol']['global_grid_step'] == .02
    for role, cfg, cfg_path in [('baseline', baseline, baseline_path), ('candidate', config, config_path)]:
        assert result['provenance'][role + '_config_sha256'] == sha(cfg_path)
        score = root / cfg['output_dir'] / 'epochs' / f'epoch{epoch:02d}-validation_scores.npz'
        assert sha(score) == audit['score_sha256'][role] == result['provenance']['score_sha256'][role]
    assert all(result['gates'][k] == v for k, v in audit['independent_gates'].items())
    labels = [f'label_{i}' for i in range(500)]
    weights = pd.read_csv(root / CONTROL / 'training-weights.csv').set_index('label').loc[labels]
    groups = support_groups(weights.train_support.to_numpy())
    table = pd.read_csv(root / result['outputs']['labels'])
    strata_path = root / audit['strata']
    strata = pd.read_csv(strata_path)
    verified = {}
    for role in ['baseline', 'candidate']:
        frame = table.loc[(table.role == role) & (table['mode'] == 'crossfit')].set_index('label').loc[labels].copy()
        assert len(frame) == 500 and frame.index.is_unique
        frame['training_group'] = groups
        verified[role] = {}
        for name in ['high', 'middle', 'low']:
            subset = frame.loc[frame.training_group == name]
            saved = strata.loc[(strata.role == role) & (strata['mode'] == 'crossfit') & (strata.training_group == name)]
            assert len(saved) == 1
            # F1 is the mean of per-seed per-label F1, never F1 of averaged counts.
            counts = prevalence_decomposition(subset, 1062)
            metrics = {**counts, 'macro_f1': float(subset.loc[subset.true_support > 0, 'f1'].mean()),
                       'macro_auc': float(subset.roc_auc.mean()), 'macro_ap': float(subset.average_precision.mean())}
            for key in ['predicted_positive_rate', 'overpredicted_labels', 'positive_excess',
                        'fp', 'fn', 'macro_f1', 'macro_auc', 'macro_ap']:
                np.testing.assert_allclose(metrics[key], saved.iloc[0][key], rtol=1e-12, atol=1e-10)
            verified[role][name] = metrics
        cf = result[role]['crossfit']
        for key in ['positive_excess', 'fp', 'fn', 'overpredicted_labels']:
            np.testing.assert_allclose(sum(v[key] for v in verified[role].values()), cf[key], rtol=1e-12, atol=1e-10)
        verified[role]['all'] = {**cf, 'macro_auc': result[role]['macro_auc'], 'macro_ap': result[role]['macro_ap']}
    evidence = {'experiment_id': task['id'], 'baseline_id': baseline['experiment_id'], 'epoch': epoch,
                'summary': path.relative_to(root).as_posix(), 'summary_sha256': sha(path),
                'audit': receipt.relative_to(root).as_posix(), 'audit_sha256': sha(receipt),
                'strata': audit['strata'], 'strata_sha256': sha(strata_path),
                'per_label_sha256': sha(root / result['outputs']['labels']),
                'score_sha256': audit['score_sha256']}
    return {'result': result, 'strata': verified, 'evidence': evidence}


def baseline_table(root):
    reference = read(root / 'artifacts/metrics/EXP-20260930-127-epoch10-analysis-summary.json')
    best127 = max(reference['history'], key=lambda x: x['mean_crossfit_macro_f1'])['epoch']
    # This historical self-comparison independently audited the best calibration epoch.
    selfpath = root / CONTROL / 'baseline-selfcheck-v2-summary.json'
    selfresult = read(selfpath)
    selfaudit = read(root / CONTROL / 'verification/baseline-selfcheck-independent-audit.json')
    assert selfresult['epoch'] == best127 and sha(selfpath) == selfaudit['source_summary_sha256']
    rows, sources = [], [selfpath.relative_to(root).as_posix()]
    for number in [127, 130, 129]:
        task = next(t for t in SUITE if t['baseline_number'] == number)
        cfg = read(root / task['baseline_config'])
        epochs = cfg['training']['epochs']
        if number == 127:
            epoch, metrics = best127, selfresult['baseline']
        else:
            results = []
            for e in range(1, epochs + 1):
                p = root / 'artifacts/metrics' / f"{cfg['experiment_id']}-epoch{e:02d}-asl-control-v2-summary.json"
                v = read(p)
                assert v['epoch'] == e and v['candidate']['experiment_id'] == cfg['experiment_id']
                assert v['provenance']['candidate_config_sha256'] == sha(root / task['baseline_config'])
                score = root / cfg['output_dir'] / 'epochs' / f'epoch{e:02d}-validation_scores.npz'
                assert sha(score) == v['provenance']['score_sha256']['candidate']
                results.append(v)
            best = max(results, key=lambda x: x['candidate']['crossfit']['macro_f1'])
            epoch, metrics = best['epoch'], best['candidate']
            sources.append(f"artifacts/metrics/{cfg['experiment_id']}-epoch{epoch:02d}-asl-control-v2-summary.json")
        assert metrics['experiment_id'] == cfg['experiment_id']
        cf = metrics['crossfit']
        rows.append([f'EXP-{number}', 'ESM-2 ' + ('150M' if number == 129 else '35M'),
                     'BCE' if number == 130 else 'ASL', 20, epochs, GROUPS[number], epoch,
                     f"{cf['macro_f1']:.6f}", f"{metrics['macro_auc']:.6f}", f"{metrics['macro_ap']:.6f}",
                     f"{cf['predicted_positive_rate']:.6%}", f"{cf['overpredicted_labels']}/500"])
    production = read(root / CONTROL / 'branch-readiness-reference103.json')
    assert production['verified_original_protocol'] and production['cpu_only']
    assert sha(root / production['score_path']) == production['score_sha256']
    old = root / 'artifacts/metrics/EXP-20260930-103-iteration8-rank-fusion-seed-aggregate-summary.json'
    assert sha(old) == production['summary_sha256']
    m = production['metrics']; cf = m['crossfit']
    rows.append(['EXP-103', 'rank融合生产参考', '—', '—', '—', '非配对基线，仅作生产回退参考', '不适用',
                 f"{cf['macro_f1']:.6f}", f"{m['macro_auc']:.6f}", f"{m['macro_ap']:.6f}",
                 f"{cf['predicted_positive_rate']:.6%}", f"{cf['overpredicted_labels']}/500"])
    cols = ['baseline_id', 'model', 'loss', 'cap', 'epochs', 'pairing_group', 'best_epoch_by_calF1',
            'calF1_at_best', 'AUC_at_best', 'AP_at_best', 'cal_pos_rate_at_best', 'overpred_labels_at_best']
    text = '# 表 1：基线速查\n\n' + markdown(cols, rows)
    text += '\n\n| 元信息 | 值 |\n| --- | --- |\n'
    text += '| 真实正例率 | 5.9166%（31417/531000） |\n'
    text += '| 校准协议 | seed 17/31/42/73/101；双向两折selection-only；逐标签精确阈值；全局0.02网格；shrinkage 25 |\n'
    text += '| 分层规则 | np.array_split(np.argsort(train_support, kind="stable"),3)；升序低/中/高=167/167/166，并列按标签顺序 |\n'
    text += '| 配对终点 | A→EXP-127 epoch6；B→EXP-130 epoch6；C→EXP-129 epoch8；峰值轮仅速查，禁止跨基线直接比 |\n'
    text += '| 计数/F1 | 过预测和excess由五seed平均逐标签预测数计算；F1为逐seed F1均值 |\n'
    text += '\n来源（项目根目录相对路径）：`' + '`；`'.join(sources + [str(CONTROL / 'branch-readiness-reference103.json')]) + '`。\n'
    return text


def result_table(root):
    five_state = root / CONTROL / 'five-state.json'
    state = read(five_state) if five_state.exists() else read(root / CONTROL / 'launch-state.json')
    statuses = {t['id']: t['status'] for t in state['tasks']}
    rows, evidence, endpoints = [], [], []
    for task in SUITE:
        endpoint = audit_epoch(root, task, task['epochs'])
        endpoints.append(endpoint)
        if endpoint:
            evidence.append(endpoint['evidence'])
        for name, chinese in [('all', '全局'), ('high', '高'), ('middle', '中'), ('low', '低')]:
            if endpoint is None:
                values = ['待审计'] * 8
            else:
                b, c = [endpoint['strata'][role][name] for role in ['baseline', 'candidate']]
                values = [f"{c[key]-b[key]:+.6f}" for key in ['macro_f1', 'macro_auc', 'macro_ap']]
                values += [f"{100*(c['predicted_positive_rate']-b['predicted_positive_rate']):+.6f}",
                           f"{int(c['overpredicted_labels']-b['overpredicted_labels']):+d}"]
                values += [f"{c[key]-b[key]:+.1f}" for key in ['positive_excess', 'fp', 'fn']]
            rows.append([GROUPS[task['baseline_number']], 'EXP-' + task['id'].split('-')[2], display_cap(task), chinese, *values])
    text = '# 表 2：五项配对结果\n\n'
    text += 'A→EXP-127 epoch6；B→EXP-130 epoch6；C→EXP-129 epoch8。所有Δ=候选−配对基线同轮；正例率Δ单位pp。\n\n'
    text += markdown(['pairing_group', 'experiment_id', 'cap_setting', 'stratum', *FIELDS], rows)
    text += '\n\n32轮和五项成功退出、全局及三层CPU审计齐备后统一归因；终点缺证据填“待审计”，不代入前一轮。\n\n'
    text += markdown(['group', 'best_cap_setting', 'main_stratum_benefiting', 'cap_is_main_cause'], [
        ['A', '待32轮统一归因', '待审计', '待判（是/否/部分）'],
        ['B', '5', '待审计', '损失×cap是否复现：待判'],
        ['C', '5', '待审计', '模型×cap是否复现：待判']])
    text += '\n\n联合判据：校准率更接近5.9166%＋过预测标签下降＋AP上升，AUC不下降（1e-12容差）。Δexcess=Σmax(五seed平均标签预测数−真值数,0)的差，不是净超额；FP/FN为五seed均值。三层按训练支持度稳定升序三等分，低/中/高=167/167/166，各层正例率分母为1062×本层标签数；逐层以自身真值率判断收敛，不统一套5.9166%。F1为逐seed逐标签F1均值，AP/AUC为原始分数宏均值。\n'
    text += '\n当前状态：' + '；'.join(f"EXP-{t['id'].split('-')[2]}={statuses.get(t['id'], '监督扩展待接入')}" for t in SUITE) + '。\n'
    text += '\n新对话入口：计划 `reports/experiments/EXP-20261002-134-135-cap-suite-extension-plan.md`；五项配置/完整ID注册 `src/esm_cap_suite.py`；实时状态 `' + (str(CONTROL / 'five-state.json') if five_state.exists() else str(CONTROL / 'launch-state.json')) + '`；独立审计 `' + str(CONTROL / 'verification') + '`；重建两表：`python -B -m src.esm_cap_tables --refresh`。\n'
    return text, evidence


def write_tables(root, refresh=False, decision_overlay=False):
    # Only these two mutable handoff views can refresh; experiment evidence is immutable.
    paths = [root / TABLE1, root / TABLE2]
    if not refresh and any(p.exists() for p in paths):
        raise FileExistsError('Tables exist; explicitly use --refresh for the handoff views')
    first = baseline_table(root)
    second, evidence = result_table(root)
    if decision_overlay:
        from .esm_cap_table_decisions import apply_decisions
        second = apply_decisions(root, second, evidence)
    for path, content in zip(paths, [first, second]):
        temp = path.with_suffix('.md.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(path)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    evidence = write_tables(Path.cwd(), refresh=args.refresh, decision_overlay=True)
    print(json.dumps({'tables_written': 2, 'result_rows': 20, 'verified_endpoints': len(evidence),
                      'submission_generated': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
