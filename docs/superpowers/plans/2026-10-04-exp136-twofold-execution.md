# EXP-136 双向两折执行续接

用户已明确授权“执行双折评估，完全按照分支计划做”，替代此前停止评估的要求。原规格为 `configs/iteration22_esm35_support_conditional_calibration_prepare.json` 及 cap 扩展计划中的条件校准部分。直接在用户指定当前分支执行，不回到旧 worktree，不自动提交或推送。

- [x] 核对当前分支、B/C 终态结论及原准备配置，保留原准备配置。
- [ ] 找到 SHA256 为 `ce1cf6e2872e35ad057532407cf897d0b69425c001d157dade6d0a4c6a476ccb` 的 EXP-127 epoch06 原始分数；验证 epoch、形状、固定 validation IDs、标签顺序、二值训练标签及划分互斥性。不得以别的分数或重训替代。
- [x] 在 `tests/test_esm_conditional_calibration.py` 添加有区别于旧实现的回归测试：先平均逐标签五 seed 预测数再统计 excess/overprediction；各原始 AP/AUC 分别一致；复用历史基线及未变层；禁用配置不可通过底层入口运行；固定ID/标签错配和训练验证交集拒绝。
- [x] 在 `src/esm_conditional_calibration.py` 修正失败测试对应实现。原始标签阈值及其并列规则复用历史方法；网格使用原 float32 0.02～0.98；先验只看 selection 折；零正例/退化层显式回退。保持原始分数不变。
- [x] 用临时合成数据完成回归测试和独立验证；合成结果不是 EXP-136 真实成绩，不登记历史结果表。
- [x] 保存新的执行配置 `configs/iteration22_esm35_support_conditional_calibration_run.json`：仅更新执行身份和状态、加入已核验 B/C 来源；输入、校准和评价合同与准备配置相同。使用未占用的 `EXP-20261004-136-esm35-support-conditional-calibration` 前缀。
- [ ] 对真实输入执行 `python -B -m src.esm_conditional_calibration --config configs/iteration22_esm35_support_conditional_calibration_run.json`。不训练、不进行测试集预测、不融合、不生成竞赛提交。
- [ ] 保存逐折先验网格、逐标签阈值/回退、逐seed全局/三层指标、逐标签五seed平均计数、原始指标一致性及历史基线预测/阈值精确复现审计。全量五seed预测仅在本地保留，拒绝覆盖所有输出。
- [ ] 独立复算输出并写结果报告与本计划的最新状态。主方案固定为只换中层先验；全三层仅诊断，不依据evaluation结果改选方案。

当前启动条件是绑定原始输入存在且哈希匹配。文件缺失时继续完成实现和检查，准确登记输入阻塞，不产生假成绩或把合成测试当作真实评估。

## 本次实际执行状态

2026-10-04 17:59（UTC+08:00），正式入口已实际调用，在绑定输入检查阶段因 `FileNotFoundError` 退出，子进程 exit code 1。没有进入 selection 阈值拟合或真实双折评价，没有产生任何新 EXP-136 指标或 run 目录。完整 stdout、stderr 和命令/目录/配置摘要保存于本地 `artifacts/runs/AUDIT-20261004-exp136-execution-readiness/`。

输入搜索覆盖 C/D/E 三盘可读位置、其他 worktree、Git 已知对象与当前远端分支，检查两个 runs.rar 和两个相关 zip，未找到绑定原始分数。测试临时目录中的同名分数是合成样本，不能替代。等待原训练机器或备份提供文件路径/可访问副本，匹配 SHA256 后继续，不重复询问启动授权，不重训替代。

定向检查命令：`python -B -m pytest tests/test_esm_conditional_calibration.py tests/test_rank_fusion.py tests/test_diagnose_esm_epochs.py tests/test_esm_epoch_diagnostics_integration.py tests/test_metrics.py -q -p no:cacheprovider`。结果48通过，13条既有退化诊断的空切片告警；本次条件校准的18项检查均通过，包括改变evaluation真值不改变对应selection折阈值，以及从本地预测独立复算F1、AP/AUC和平均计数口径。

状态报告：[EXP-136执行与输入阻塞](../../../reports/experiments/EXP-20261004-136-conditional-calibration-execution.md)。评估目标仍未完成，不能将执行器准备完成写成实验已完成。

## 2026-10-05 新提供输入的核验状态

用户提供仓库根目录下的 `epoch06-validation_scores.npz`。epoch、1062×500形状、固定ID及标签顺序均通过，但SHA256为 `d0e81890f1ec54c18cd92a3e8cd488b13a5e73535fa1e65a3861275c97e1a6d4`，与绑定hash不同；只读复算AUC/AP/默认0.5 F1及全部500标签的分数统计也与归档EXP-127 epoch06不同。不是仅有压缩容器差异，原始输入验收项保持未完成。

核验记录：`artifacts/metrics/AUDIT-20261005-exp136-supplied-input-summary.json`。未改绑定合同，未拟合阈值，未产生EXP-136结果；用户提供文件保留原位且内容未改。等待确认文件来源和匹配的原始副本，不重新询问启动授权。

2026-10-05后续新增EXP-134压缩包后，文件身份已确认：根目录epoch06与包内EXP-134 epoch06逐字节相同，且匹配历史EXP-134来源。完成该候选六轮原始指标/来源和终点历史校准复核，见 `artifacts/metrics/AUDIT-20261005-exp134-restored-summary.json` 及[核验报告](../../../reports/experiments/2026-10-05-exp134-restored-input-audit.md)。仍缺正确EXP-127原始分数；上面的EXP-136输入验收/真实执行/结果复核项目保持未完成。EXP-134复算属于历史证据审计，没有拟合新的条件先验。

2026-10-05推送前全量检查：条件校准端到端测试改为独立CPU子进程，避免其他GPU测试导入torch造成测试相互干扰；新增torch已导入时正式入口拒绝执行的检查，正式算法和保护不变。全量242项通过，13条既有退化诊断告警。
