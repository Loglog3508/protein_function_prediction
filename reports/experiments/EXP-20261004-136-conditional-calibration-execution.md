# EXP-136 条件校准双向两折：执行与输入状态

状态：**已获启动授权，执行器检查通过；用户提供的文件未通过绑定输入核验，真实条件校准评估尚未开始。** 2026-10-05最新核验见文末。没有EXP-136真实成绩，未登记迭代结果表，不能依据合成测试宣布条件校准有效。

当前分支 `codex/esm-labelwise-iteration6`，基于 `a461a99`。用户明确要求“执行双折评估，完全按照分支计划做”，替代之前停止双折评估的指令，也授权完成原计划要求的条件校准执行器。原准备配置保持不变，新的执行配置为 `configs/iteration22_esm35_support_conditional_calibration_run.json`，身份及输出前缀为 `EXP-20261004-136-esm35-support-conditional-calibration`；不是旧 EXP-091 融合评估。

## 固定协议

- 输入严格绑定EXP-127 epoch6，1062×500，SHA256 `ce1cf6e2872e35ad057532407cf897d0b69425c001d157dade6d0a4c6a476ccb`。
- 固定tail训练/验证IDs和标签顺序；保持原seed42划分。分层只用训练支持度，稳定升序低/中/高167/167/166。
- 五seed17/31/42/73/101，双向两折，阈值和本层先验只用selection折；原float32 0.02网格及并列规则、逐标签精确阈值、shrinkage25保持一致。
- 配对基线为全局先验，主方案只换中层先验，高/低层阈值精确复用；全三层替换仅作预指定诊断，不能按evaluation结果重新选主方案。
- 原始分数不转换，AP/AUC分别保持一致。F1平均逐seed逐标签F1；预测数先按逐标签五seed平均，再计算overprediction和excess；同时报告TP/FP/FN及各层自身真值率。
- CPU评估，不训练，不预测测试集，不融合，不生成提交，不覆盖历史产物，不自动提交/推送。

执行器在运行前比较准备配置的input、stratification、calibration、evaluation四个完整字段，拒绝协议漂移；核对持久A/B/C终态结论及来源hash。新的启动授权已满足，当前等待的是数据，不是再次批准。

## 已完成实现检查

修正此前实现把逐seed excess/overprediction取平均的口径；修正AP/AUC跨方案一致性检查；补充固定IDs顺序、标签顺序、epoch、分数hash、训练验证互斥、二值标签、输出独占及所有入口的启用检查。

基线对每个seed调用历史 `src.diagnose_esm_epochs.crossfit_predictions`，逐元素核对预测与阈值；主方案高/低层再作精确一致检查。保存selection先验网格、逐标签阈值/回退、逐seed指标、逐标签平均计数、配对判据和验证回执；完整预测只保存在本地run目录，用于独立审计。生产数据还必须复现EXP-128历史epoch6的AUC、F1、预测率和计数基线。

定向检查共48项通过，涵盖本执行器及历史rank阈值、epoch诊断和指标实现。其中本执行器18项合成检查通过，包括evaluation标签扰动不能改变对应selection折阈值，以及不通过执行器的汇总函数、直接从预测独立计算F1/平均计数/AP/AUC。13条告警来自既有诊断测试的退化空切片。本次没有修改历史校准算法或掩盖计算异常。

## 实际入口结果

2026-10-04 17:59:17～17:59:19（UTC+08:00），在当前仓库目录实际执行：

```shell
python -B -m src.esm_conditional_calibration --config configs/iteration22_esm35_support_conditional_calibration_run.json
```

子进程退出码1，失败阶段为绑定输入检查：

```text
FileNotFoundError: saved score matrix not found:
artifacts/runs/EXP-20260930-127-esm35-last4-full-epoch10/epochs/epoch06-validation_scores.npz
```

完整命令、实际目录、配置SHA256、退出码和stdout/stderr保存于忽略目录 `artifacts/runs/AUDIT-20261004-exp136-execution-readiness/`；未进入真实双折拟合，没有该实验的预测或指标文件，也没有训练启动。

C/D/E三盘可读位置、现有worktree、Git已知对象、当前远端分支和相关压缩包中均未找到目标输入。同名测试文件是合成矩阵，现有旧实验产物不满足绑定合同，不能替代。已有EXP-128统计也不能反推出完整逐样本分数。

## 续接

需要原训练机器或备份中的原始epoch06文件路径或可访问副本。取得后先核对SHA256并恢复到绑定路径，再直接运行已授权命令；如hash不符、ID/标签/epoch不符或历史基线不复现，保留错误并停止，不绕过检查、不更换基线、不重训替代。新的评估结果全部使用预留的新前缀，完成后独立审计并报告主方案与诊断方案的全局及三层权衡。

## 2026-10-05 用户提供文件后的核验

用户已将 `epoch06-validation_scores.npz` 保存到当前仓库根目录。该文件存在，大小1,148,047字节；epoch=6、float32分数形状1062×500、固定validation IDs及标签顺序完全一致，分数均为有限的[0,1]概率。但文件SHA256为 `d0e81890f1ec54c18cd92a3e8cd488b13a5e73535fa1e65a3861275c97e1a6d4`，与计划绑定的 `ce1cf6e2872e35ad057532407cf897d0b69425c001d157dade6d0a4c6a476ccb` 不同。

只读复算使用相同validation IDs对应的训练表真值，并核对归档EXP-128逐标签真值支持数全部相同。无需拟合阈值即可观察到以下内容差异：

| 指标 | 提供文件 | 归档EXP-127 epoch06 | 差值 |
| --- | ---: | ---: | ---: |
| 原始Macro AUC | 0.767674507658 | 0.768011665241 | -0.000337157583 |
| 原始Macro AP | 0.176444779311 | 0.172024943671 | +0.004419835641 |
| 默认0.5阈值Macro F1 | 0.166653250355 | 0.176903759004 | -0.010250508649 |

500个标签的AUC、AP、分数均值/最小值/最大值分别均与归档不同（绝对容差1e-12）；归档summary和逐标签CSV的SHA256与EXP-128验证回执一致。因此差异影响原始分数内容，不能仅由NPZ重新压缩解释；尚不能确认它属于哪个实验或哪次预测。

核验输出使用独立前缀 `artifacts/metrics/AUDIT-20261005-exp136-supplied-input-summary.json`；本地只读辅助为 `artifacts/runs/AUDIT-20261005-exp136-supplied-input/check_supplied_input.py`。未移动、修改或重新封装用户文件，未更改准备配置及绑定hash，未拟合校准阈值，未产生EXP-136输出。本次未启动正式入口，避免使用已知不匹配输入；原2026-10-04缺文件的启动记录仍作为历史保留。

当前阻塞从“用户尚未提供文件”更新为“提供文件与指定基线不符”。已询问文件来源；需要确认原始EXP-127 epoch06副本。启动授权继续有效，取得匹配文件即可按原计划运行。

## 2026-10-05 新增EXP-134后：输入身份已确认

新提供的EXP-134压缩包、解压后的epoch06及仓库根目录epoch06哈希完全相同，均匹配原归档EXP-134候选来源。已独立核验六轮hash/原始指标，并复现epoch06五seed历史校准；此前文件身份不明的疑问已解决，根目录文件是EXP-134的分数。

详见[EXP-134原始文件恢复核验](2026-10-05-exp134-restored-input-audit.md)，新摘要 `artifacts/metrics/AUDIT-20261005-exp134-restored-summary.json`。这补齐部分cap排查原始证据，但没有提供EXP-136绑定的EXP-127分数；不得把EXP-134改名或修改输入hash后冒充原计划实验。继续等待正确EXP-127副本，授权和固定协议均不变。

## 2026-10-05 推送前全量检查

全量pytest首次运行因其他GPU测试在同进程导入torch，使4项条件校准测试触发正式执行器的“必须使用未导入torch的新CPU进程”限制。修改条件校准端到端测试为使用当前环境解释器的独立子进程，并新增已导入torch时必须拒绝执行的回归检查；同时确认独立进程结果记录torch未导入。没有放宽正式执行器检查或改变科学协议。

修正后 `python -B -m pytest -q -p no:cacheprovider --tb=short` 全量 **242 passed，13 warnings**；告警仍来自既有退化诊断空切片。EXP-136输入缺失和未执行的状态保持不变，代码检查通过不代表实验已经完成。
