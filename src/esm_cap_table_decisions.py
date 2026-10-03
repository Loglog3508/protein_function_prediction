"""Overlay verified user decisions without changing frozen cap-run evidence."""
from __future__ import annotations

from pathlib import Path

from . import esm_cap_tables as tables

DECISIONS = Path('configs/esm_cap_attribution_decisions.json')


def apply_decisions(root, content, evidence):
    policy_path = root / DECISIONS
    if not policy_path.exists():
        return content
    policy = tables.read(policy_path)
    assert policy['schema_version'] == 1
    assert not policy['conditional_calibration_started']
    verified = {int(item['experiment_id'].split('-')[2]): item for item in evidence}
    assert set(verified) == {131, 132, 133, 134, 135}
    for group in ['A', 'B', 'C']:
        decision = policy[group]
        assert decision['state'] == 'confirmed_for_completed_suite'
        for expected in decision['evidence']:
            actual = verified[expected['experiment_number']]
            assert actual['epoch'] == expected['epoch']
            assert actual['summary_sha256'] == expected['summary_sha256']
            assert actual['audit_sha256'] == expected['audit_sha256']
    placeholders = {
        'A': '| A | 待32轮统一归因 | 待审计 | 待判（是/否/部分） |',
        'B': '| B | 5 | 待审计 | 损失×cap是否复现：待判 |',
        'C': '| C | 5 | 待审计 | 模型×cap是否复现：待判 |',
    }
    for group, old_row in placeholders.items():
        decision = policy[group]
        new_row = '| ' + group + ' | ' + ' | '.join(decision[key] for key in [
            'best_cap_setting', 'main_stratum_benefiting', 'cap_is_main_cause']) + ' |'
        assert content.count(old_row) == 1
        content = content.replace(old_row, new_row)
    old_note = '32轮和五项成功退出、全局及三层CPU审计齐备后统一归因；终点缺证据填“待审计”，不代入前一轮。'
    assert content.count(old_note) == 1
    content = content.replace(old_note, '32轮终态已完成；A/B/C均已用配对终点和全局/高/中/低审计归因，终点缺证据不代入前一轮。')
    content += '\n终态判据：B→' + policy['B']['criterion'] + '；C→' + policy['C']['criterion'] + '。历史四项联合gates保留原定义。\n'
    content += '\n后续：' + policy['next_branch'] + '。条件校准配置已准备、未启动：`' + policy['conditional_calibration_config'] + '`。\n'
    content += '\n机制边界：' + policy['mechanism_note'] + '。\n'
    content += '\n结论持久来源：`' + DECISIONS.as_posix() + '`；两种刷新入口均会校验并保留 A/B/C 终态判断：`python -B -m src.esm_cap_tables --refresh` 或 `python -B -m src.esm_cap_table_decisions --refresh`。\n'
    return content


def install_overlay():
    if getattr(tables.result_table, '_decision_overlay', False):
        return
    original = tables.result_table

    def with_decisions(root):
        content, evidence = original(root)
        return apply_decisions(root, content, evidence), evidence

    with_decisions._decision_overlay = True
    tables.result_table = with_decisions


def main():
    tables.main()


if __name__ == '__main__':
    main()
