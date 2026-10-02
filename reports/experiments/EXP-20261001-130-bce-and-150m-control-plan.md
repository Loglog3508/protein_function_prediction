# EXP-130 BCE 优先与 EXP-129 150M 对照执行计划

## 用户授权与执行顺序

用户在 EXP-128 诊断后明确授权启动两项对照，覆盖此前“150M 仅准备”“不启动损失对照”的限制。单卡 8GB 环境为避免训练互相竞争与显存干扰，先运行 EXP-130 35M BCE 六轮，再运行原准备的 EXP-129 150M ASL 八轮；后者排队不等于已经启动。进程、退出码和逐轮分数是执行状态依据，不仅看 GPU 使用率或旧状态文件。

禁止启动池化、逐标签独立头或其他实验；不生成竞赛提交，不进行测试集预测，不自动 Git 提交/推送。

## EXP-130：损失机制单变量

配置 `configs/iteration19_esm35_bce_full_epoch6.json`，相对 EXP-127 除实验身份/路径和 epochs=6 外，只把损失对象从 ASL 换为 `{"name":"bce"}`。seed42、完整训练/验证 IDs、模型35M及revision、最后4层解冻、mean_max池化、1022窗口/128 overlap、batch8、累积2、head/encoder LR、AMP、每轮保存分数等不变。

“纯 BCE”在本对照中指不带非对称聚焦/概率裁剪的 `binary_cross_entropy_with_logits`，**保留 EXP-127 的正类权重计算及 cap=20**。同时取消正类权重会成为第二个变量，不在本次授权单变量范围内。不改变 checkpoint_metric，保持 macro_f1；三项新判据在离线分数评估中报告，避免改选模协议。

## EXP-129：150M原配置

直接使用 `configs/iteration18_esm150_last4_full_epoch8.json`，不改变预先核验的模型revision、8轮、每轮保存分数及其他参数。模型架构变化伴随最后4层参数量/head输入维度变化；只在共同 epoch1～8 上与 EXP-127 配对，不用35M第10轮对150M第8轮冒充等训练预算。

## 共同判据与报告

每轮保存原始验证分数，按 EXP-127 的相同 IDs/标签顺序复算：

1. 连续分数 Macro ROC AUC。
2. Macro AP；这是整体 precision-recall 检索指标，不是单一 top-K 的精度，也不能直接推断所有高分截断。
3. 默认0.5与既有五 seed交叉拟合校准的预测正例率、相对真实正例率倍数、过预测标签比例、正例超额及TP/FP/FN。
4. 校准协议保持 seed17/31/42/73/101、精确逐标签阈值、全局0.02网格、shrinkage25，不搜索新参数。

EXP-130 主判据是 **AP 高于 ASL 同轮，并且校准过预测下降**，同时报告AUC是否牺牲；不用训练loss跨损失直接比大小。EXP-129 同时报告 AUC/AP/过预测，不因仅AUC增加宣布整体胜出。所有判断基于分数，不先登记成绩；部分轮次与完整6/8轮严格区分。

## 运行与恢复约束

- 激活用户现有 GPU 环境，使用 active `python`；不在配置或可提交源码中固化机器解释器/缓存路径。
- 后台监督进程仅串行执行这两项已授权命令，留 stdout/stderr、真实子进程PID、开始时间、配置hash、退出码及后继启动记录；窗口隐藏。
- 启动前拒绝已有输出目录或重复正在运行的同配置任务，不覆盖历史模型/指标。监督进程退出后的锁/状态文件不能单独证明运行中。
- 失败（含 OOM）不自行减batch、改梯度累积、改窗口或改AMP，也不从头静默重启；记录确切失败，下一项状态必须说明。
- 150M需实际子进程启动后才能记 `running`；仅排队期间记 `queued`。任务仍活跃，待两项启动并完成所需对照评估后再核验最终结论。

具体本机PID、路径及日志回执仅保留在被忽略的本地控制目录。

## 初始执行证据

EXP-130 实际启动记录为 `2026-10-01T12:27:14.0339770+08:00`（运行机器时区），监督回执为 `artifacts/runs/EXP-20261001-130-controls/launch-state.json`。随后已核验训练 Python 及其监督进程存活、实际加载35M权重且GPU训练活跃；首次 epoch 尚未完成，因此不登记AP/AUC成绩。EXP-129当前仅排队，不写作“已启动”。

CPU 评估入口为 `src.compare_esm_controls.py`，后台读取各轮已发布 history 后再评估保存分数，确保不读取半写入 NPZ；输出每轮独立 `*-epochNN-asl-control-summary.json`、逐标签和seed明细，拒绝覆盖。持续观察器不导入torch，不占GPU；完成全部6/8轮分数及三项同轮评估后才写完成回执。

评估器启动回执为 `artifacts/runs/EXP-20261001-130-controls/observer-launch.json`；实际 stdout 已输出 `watcher_status=ready, cpu_only=true, torch_imported=false`，对应 Python 存活且 stderr 为空。配置/损失/评估/ESM训练器目标测试 **24 passed、45 subtests passed**。另直接复用 EXP-127 的 ep6 与自身配对做 CPU 自检，AUC/AP/校准正例率/过预测标签比例/正例超额/F1 的差值全部为零，历史 F1/正例率一致；自检只在本地控制目录保存，不把它登记为 BCE 或150M结果。

截至本次启动核验，BCE第1轮仍训练中，没有已发布 epoch history 或新轮分数；150M运行目录不存在，故它仍为queued。监督和CPU观察进程均真实存活；目标保持进行中，不能因配置准备完毕或任务进入队列便宣布两项对照完成。

## 后续：首轮发布与CPU审计修复

BCE epoch1已发布，阶段结果见 `EXP-20261001-130-bce-epoch6-results.md`；原“首轮尚未发布”仅为启动时的历史记录。审计修复默认/单seed无符号计数下溢，只替换CPU评估观察器，不重启GPU训练或改变配置；旧指标保留，后续及首轮重算统一使用`-asl-control-v2`产物。当前CPU观察器回执改为本地 `observer-v2-launch.json`，日志为 `observer-v2.stdout.log`，不能依据旧observer PID判断当前运行。AP/AUC、校准F1/正例率与原首轮差值经独立核验不变；六轮训练尚未完成，150M尚未启动。
