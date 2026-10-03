# 表 1：基线速查

| baseline_id | model | loss | cap | epochs | pairing_group | best_epoch_by_calF1 | calF1_at_best | AUC_at_best | AP_at_best | cal_pos_rate_at_best | overpred_labels_at_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-127 | ESM-2 35M | ASL | 20 | 10 | A | 6 | 0.214033 | 0.768012 | 0.172025 | 12.043540% | 472/500 |
| EXP-130 | ESM-2 35M | BCE | 20 | 6 | B | 6 | 0.211316 | 0.770315 | 0.171334 | 12.520527% | 481/500 |
| EXP-129 | ESM-2 150M | ASL | 20 | 8 | C | 8 | 0.218878 | 0.773379 | 0.177008 | 11.882373% | 480/500 |
| EXP-103 | rank融合生产参考 | — | — | — | 非配对基线，仅作生产回退参考 | 不适用 | 0.338248 | 0.824492 | 0.298196 | 8.882034% | 467/500 |

| 元信息 | 值 |
| --- | --- |
| 真实正例率 | 5.9166%（31417/531000） |
| 校准协议 | seed 17/31/42/73/101；双向两折selection-only；逐标签精确阈值；全局0.02网格；shrinkage 25 |
| 分层规则 | np.array_split(np.argsort(train_support, kind="stable"),3)；升序低/中/高=167/167/166，并列按标签顺序 |
| 配对终点 | A→EXP-127 epoch6；B→EXP-130 epoch6；C→EXP-129 epoch8；峰值轮仅速查，禁止跨基线直接比 |
| 计数/F1 | 过预测和excess由五seed平均逐标签预测数计算；F1为逐seed F1均值 |

来源（项目根目录相对路径）：`artifacts/runs/EXP-20261002-131-controls/baseline-selfcheck-v2-summary.json`；`artifacts/metrics/EXP-20261001-130-esm35-bce-last4-full-epoch6-epoch06-asl-control-v2-summary.json`；`artifacts/metrics/EXP-20261001-129-esm150-last4-full-epoch8-epoch08-asl-control-v2-summary.json`；`artifacts\runs\EXP-20261002-131-controls\branch-readiness-reference103.json`。
