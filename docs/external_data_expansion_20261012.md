# 外部蛋白数据扩展：有效监督、来源与实施协议

研究日期：2026-10-12。范围：核对官方数据入口和维护者 GitHub，结合本地 EXT379 资产提出下一批数据方案。本次未启动大规模下载或训练。

## 结论

下一批优先考虑 **GOA 的 UniProtKB 全库注释（覆盖 Swiss-Prot 与 TrEMBL）及独立分层的 IBA 注释**，其次是 GO Consortium 的物种专家数据库注释；UniRef / RefSeq 用作补充表征与检索语料。它们都必须先与现有资产做完整序列和证据差集，不能用下载包大小或注释行数承诺新增训练蛋白数量。[S1–S7]

F1 的数学最大值已经是 1。增加数据可能提高当前条件下可达到的泛化性能，但不保证提升，也不自动消除标签冲突。当前外部样本的 500 匿名主任务标签全部未知：新增序列要通过 GO 辅助多任务、合适的阳性/未知（PU）目标或表征学习影响主任务；普通的 500 标签监督损失无法直接使用这些新序列。没有权威映射时，GO ID 与 `label_0`–`label_499` 的对应关系不能靠猜测生成。[L1]

## 本地基线与真实增量

EXT379 保留全部 113,796 条官方原始训练记录，加入 13,522 条外部唯一序列，共 127,318 条。Swiss-Prot 原始快照扫描了 575,748 条 reviewed 记录；所以当前严格证据政策下，简单再下载同一 Swiss-Prot 包已经没有相同口径的未提取增量。原报告记录其 Last-Modified 为 2026-09-03，使用 GO `releases/2026-07-26`；这些是已有快照，不能表述为本次新增下载。[L1]

2026-10-12 本轮对 EXT379 产物的只读复核：

| 指标 | 数量 | 含义 |
|---|---:|---|
| 外部序列新增官方 500 标签已知单元格 | 0 | 外部 `main_known` 全为 0 |
| 官方记录 GO 阳性单元格 | 2,957,909 | 包含本体传播后的辅助监督 |
| 外部记录 GO 阳性单元格 | 180,295 | 不等于直接实验次数 |
| 官方记录覆盖 GO 项 | 28,292 | GO 词表覆盖 |
| 外部记录覆盖 GO 项 | 10,562 | 多数与官方 GO 监督覆盖重合 |
| 仅外部记录覆盖 GO 项 | 569 | 辅助监督增加的 GO 覆盖，不是新增匿名标签 |
| 外部支持数为 1 的 GO 项 | 4,285 | 统计外部序列支持，不能当成 500 标签频率 |
| 外部支持数至少 20 / 至少 100 的 GO 项 | 1,034 / 258 | 支持数可能受本体祖先传播影响 |
| 外部直接证据唯一 accession | 13,688 | 合并完整相同序列后为 13,522 条 |
| 官方完全同序列组 | 2,613 | 保留原始行 |
| 其中主标签向量冲突组 / 涉及行 | 2,171 / 4,676 | 同一序列可能对应不同的 500 标签向量 |

复核输入：EXT379 的 `canonical-sequences.csv`、`main-labels.npz` 和 GO CSR 数组；使用现有加载数据做统计，没有训练。完全同序列标签冲突意味着仅以序列为输入的确定性预测无法同时匹配这些行的全部真值；它不提供一个已计算的 Macro F1 数值上限。增加别处的序列不能直接解除原始冲突，也不能据此删去原始官方行。

## 三类实质可行的数据源

### 1. GOA UniProtKB 全库：首先做差集

GOA 多物种 README 明确覆盖 UniProtKB/Swiss-Prot 和 UniProtKB/TrEMBL，也包含 RNAcentral 的 RNA 与 Complex Portal 的复合物。读取 `goa_uniprot_all.gaf.gz` 时，至少限制数据库为 `UniProtKB`、实体类型为蛋白，并核对 isoform 字段；GAF 是注释表，序列还需要按 accession 从同版本 UniProtKB 取得。TrEMBL 身份本身不等于所有 GO 注释都是实验证据，也不等于所有注释都可靠：应逐条依据 evidence 和 reference 过滤。[S1, S2, S4]

首批仍保留 EXT379 的实验类 `EXP/IDA/IPI/IMP/IGI/IEP/HTP/HDA/HMP/HGI/HEP` 与单列的 `TAS/IC`。另建 **IBA 层**：GO 官方说明 IBA 属于系统发育推断，基于有实验支持的祖先功能模型且经过人工审阅；它可增加覆盖，但不是该蛋白本身的直接实验。`ISS/ISO/ISA/ISM/IGC/RCA` 如采用，也各自保留来源与分层。`IEA` 为未逐条人工审阅的电子注释，不能混入强监督；`ND` 不提供已知功能阳性；含 `NOT` 的行不作阳性。[S3]

实时访问官方 FTP 目录观察到：`goa_uniprot_all.gaf.gz` 展示大小 **27G**、目录修改时间 `2026-10-11 22:11`；`goa_uniprot_all.gpa.gz` 为 **6.8G**。但 `current_release_numbers.txt` 同时写 `uniprot 236 2026-07-28`，部分 gcrp 文件也仍显示 7 月日期。目录和 manifest 不一致，**未核实当前大文件对应的语义 release**；修改时间不能代替注释 header 的版本。因此本次不把它称为已确认的最新 release，也不直接下载整个大包。[S1a, S1b]

获取建议：先选和训练序列物种/蛋白家族分布有关的 GOA proteome 或 species 子集，并分别统计严格层与 IBA 层的 accession 和 GO 差集；需要完整覆盖时，再固定已确认发布版本，流式解析压缩文件、仅保留目标实体和证据、分批取得 accession 对应序列。获取子集只是减少首批评估成本，不能宣称已经穷尽全库。GOA README 允许复制和分发，但要求随副本保留版权声明；UniProt 数据使用 CC BY 4.0。[S1, S2, S5]

### 2. GO Consortium / MOD：补专家证据和来源

GO 官方推荐常见模式物种使用专家维护的物种数据库（MOD）注释，包含直接实验以及其他证据类型，支持 GAF 和 GPAD/GPI。优先按本地物种分布选择 SGD、FlyBase、WormBase、MGI 等提供方，而非默认人类数据最合适。统一处理序列 accession、isoform、taxonomy、evidence、reference、`Assigned_By` 与注释日期；基因 ID 必须通过可追溯映射得到相应蛋白序列，不能仅按名称关联。[S2, S4]

这类数据与 GOA 可能重复；其收益也可能是现有序列上更完整的注释，而非新序列。要分别汇报新增序列、新增直接证据和既有序列新增 GO 阳性，跨库同一 accession / GO / reference / evidence 的证据不重复计为独立实验。GO 官方要求使用匹配本体版本并记录 GAF header 日期；推荐固定带日期的月度 release / DOI。GO 数据产品为 CC BY 4.0，软件仓库另按各自许可。[S2, S4, S6]

GO 团队的 2026 年迁移公告将原 `/annotations/goa_human.gaf.gz` 等入口迁移为 `/annotations/gaf/HUMAN-uniprot.gaf.gz`、`DROME-mod.gaf.gz`，并新增 `/gpad`、`/gpi` 分类目录；有 MOD 的物种可能提供 MOD-ID 与 UniProt-ID 两种视图，两者不能重复计作两份独立标注。公告说明新文件合并各提供方的人工注释与 IEA，因此即使文件名含 `mod`，也仍要逐行 evidence 筛选。使用官方映射表选择新 URL，再检查实际文件 header / release，而非假设旧入口还可用。[G4]

### 3. UniRef / RefSeq：表征和检索补充

UniRef 把近似序列组织为 100%、90%、50% 身份阈值的簇，用于减少检索冗余；不是新的 500 标签真值来源。首选与任务相关的簇代表或物种子集，作为检索库、表征语料或去重索引。仅用代表序列也不能保证下游验证同源簇已隔离，仍需对官方 train/validation/test 建立明确的相似性检查。[S7]

RefSeq 有蛋白、转录本和基因组；记录由计算、人工整理、协作和其他来源共同生成。`XP_` 等模型蛋白是预测产品；`NP_` 或 reviewed 状态也不能替代每条功能注释的实验证据。GO 下载指南说明 RefSeq 的 GO 文件面向有 GCF_ 参考组装的真核基因组，使用 NCBI Gene ID，需要 Gene→protein / isoform 映射；不把所有 RefSeq 蛋白描述当 GO 金标准。按目标物种/组装获取 FASTA 与元数据，避免全量下载，再独立考虑自动 GO 弱监督。[S2, S8]

UniProt 包及 UniRef 适用 CC BY 4.0。NCBI 对分子数据本身不设置使用或分发限制，但同时说明原始提交者可能主张专利等权利，不能写成无条件的统一 CC0 授权。保留来源与原始许可说明。当前没有下载 UniRef 或 RefSeq 包，精确字节数和新增序列数均待选定版本/子集后核实。[S5, S9]

## GitHub 候选：先找来源，再判断增量

GitHub 适合发现维护者的数据准备脚本和快照入口；代码库自身不是更多已标注蛋白。2026-10-12 通过 GitHub API 读取以下维护者 README，并查询 TALE / DeepGraphGO 的完整文件树：[G1–G3, G6, G7]

| 项目 | README 确认的入口和来源 | 对本项目的判断 |
|---|---|---|
| DeepGOPlus | `bio2vec.net/data/deepgo/`；包括运行包、CAFA3 与 2016 比较集；更新流程重新读取 GO / Swiss-Prot | 可能复用 Swiss-Prot，需差集；优先借鉴解析/建模，不能当新全库 |
| DeepGO-SE（仓库 deepgo2） | `bio2vec.net/data/deepgo/deepgo2/training-data.tar.gz`；Swiss-Prot 与 neXtProt 评估集；生成流程指定 `uniprot_sprot.dat.gz` | neXtProt 候选单独核对；作者原划分不能替代本项目验证隔离，需查证据、版本与实际重合 |
| DeepGOZero | `deepgo.cbrc.kaust.edu.sa/data/deepgozero/`；Swiss-Prot `21_04` 与 NetGO2.0 基准包；另外给出零样本预测文件 | 旧快照可能大幅重合；模型输出是预测，不是 gold GO 证据 |
| TALE（Shen-Lab/TALE） | 仓库实际包含 `data/CAFA3/` 和 `data/ours/` 的 train/test sequence、label 和 ontology 文件；README 的序列记录含 Swiss-Prot ID、accession、日期、序列、GO | 可作为较小的首批差集候选；MF/BP/CC 之间序列会重合，GO 索引需用其本体表转换，不能作为匿名 500 标签直接拼接 |
| DeepGraphGO（yourh/DeepGraphGO） | 仓库实际包含 `data/data.z01` 至 `.z06` 与 `data.zip`；数据 README 描述 protein–GO pairs、PPI / InterPro 矩阵及 `ppi.fasta` 对应索引 | 有真实分卷数据资产，但当前未解包，序列、标签证据和版本需核实；网络节点数不能当有标签蛋白数 |

本次对三条维护者下载入口作 HTTP HEAD 核验，均返回 200，没有下载其正文：

| 入口 | Content-Length（字节） | 服务器 Last-Modified | 说明 |
|---|---:|---|---|
| DeepGO-SE `training-data.tar.gz` | 7,062,285,322 | 2023-10-25 17:47:59 GMT | 约 7.06 GB；含数据与特征/图，大小不等于序列数 |
| DeepGOPlus `data-cafa.tar.gz` | 1,072,198,260 | 2021-06-02 13:11:04 GMT | 约 1.07 GB；README 标注 CAFA3 |
| DeepGOPlus `data-2016.tar.gz` | 758,972,403 | 2019-04-16 11:18:20 GMT | 约 0.76 GB；历史比较集 |

DeepGraphGO 文件树所示七个分卷合计 563,503,656 字节，约 0.56 GB；TALE 的 `data/ours/train_seq_mf` 为 25,716,178 字节，`train_seq_bp` 为 42,467,876 字节，`train_seq_cc` 为 35,214,105 字节。这里只核实目录资产及入口可访问性，未证明下载完整性或准入后的新增量。Last-Modified 是服务器文件时间，不是数据库版本或注释时间。首批可先核实 TALE 中一份序列表及对应 GO 表的来源和差集，再决定是否获取较大的包。

本次未下载上述数据包或确认其完整 contents、数据条款及证据筛选。README 指向数据不等于整个数据包已经通过准入。后续必须冻结 Git commit、包 checksum、原始数据库 release、GO version，以及每条保留注释的证据。作者公开的 test 集并非自动禁止作为其他任务训练语料，但只能在本项目规则、注释时间窗口与验证/测试隔离成立时使用；作者原有 train/valid/test 划分不能代替本项目划分。仓库软件许可证不能自动覆盖其外链数据。

GO Consortium 官方 GitHub 发布流水线与迁移映射提供数据格式和来源信息，适合用于可靠的读取器设计；它们仍是 GO / GOA 数据的生成基础，不是额外独立 gold 数据库。[G4, G5] CAFA 基准包可作为后续候选，但 DeepGOPlus README 中的 CAFA3 包是作者提供的基准入口，不能直接称为本竞赛的官方新增训练集；社区镜像的上传者身份、原始发行源、标签日期和与本项目官方数据的重合均需另外验证。本次未合并 CAFA 数据。

## 实施协议与验收

1. **保留全部原始官方记录。** 保持固定 seed42 验证 ID。新 EXT 编号、输出目录及实验指标前缀必须唯一，禁止覆盖 EXT379 或已有指标。新增源可按覆盖收益排序，不按频率裁剪官方主表。
2. **先冻结版本再提取。** Receipt 至少含 source URL、UTC 获取时间、数据库 release、GAF/GPAD format 和 header、注释生成日期、本体/ECO 版本、SHA-256、提供方 checksum、原始 license。若 manifest 与 header 不符，先记录不一致并选可追溯历史版本，不能靠 FTP 修改时间猜版本。
3. **证据分层保留。** 原始行留存，解析 protein / isoform / accession / taxonomy / GO / relation / NOT / evidence / reference / With-From / Assigned-By / annotation extension / date。按 GAF header 选择 2.1 或 2.2 解析器；GPAD 的 ECO code 通过冻结映射处理，不能直接当作 GAF 字符串 evidence。
4. **完整序列去重并单独保留证据。** 使用本项目已验证的序列编码还原，按完整真实氨基酸序列 SHA-256 建主键；保留多个 accession 和不同证据。同序列新增证据有价值，但不计作新蛋白。GAF 每行是 gene product–GO association，同一蛋白会有多行。[S4]
5. **隔离验证和测试。** 外部序列至少排除与官方 train/test 相同的序列。历史对照保持 seed42 的 train/validation ID，并将验证 ID 及其完整相同序列的全部 GO 监督、外部证据和拟合用检索记录隔离。另设同源簇隔离评估，阈值、覆盖度和方法预先固定；不要把完全相同序列去重写成同源隔离。测试序列只可用于去重检查；测试标签或基于测试答案获得的 GO/匿名标签不可用于训练。
6. **未知保持未知。** 外部匿名标签 `main_known=0`；GO 未注释项不能填负。`NOT` 需保留语义和注释上下文，只有经过单独规则核实才可形成特定 GO 的已知阴性；不能推广为整条序列或所有后代都阴性。保留直接证据和 `is_a` / `part_of` 传播区别。
7. **先量化准入后的增量。** 每层汇报注释行数、唯一 accession 数、完整唯一序列数、与 EXT379 重合数、验证/测试排除数、新序列新增 GO 阳性、新覆盖 GO 项、每项独立序列支持，以及预计压缩/解压/特征磁盘和处理耗时。数字未知时写未知，不写估计成承诺。
8. **小规模验证学习收益。** 先证明现有 GO 辅助层能实际参与优化：在固定验证隔离下比较官方基线、现有 EXT379、严格新增层、严格+IBA 层。阳性/未知目标需有避免全阳性的辨别机制；GO 辅助的收益最终以 500 标签 Macro F1 验证，不能用 GO 项数量代替。IEA 弱监督作为另一个实验，不混写实验金标准。

上述为数据研究与内部实验方案；实际参赛使用外部数据及注释日期窗口，还需对应本竞赛原始规则。现有资料没有完整规则正文，本次不宣称已确认参赛许可，也不因此阻止只读研究。

## 来源

访问日期均为 2026-10-12。Web 浏览工具未返回可检查正文，因此额外通过官方 HTTPS / GitHub API 读取小体积页面核验；未下载大数据文件。目录数字为访问时观察，下载前必须复核。

- [L1] 本地基线：[官方原始文件与外部数据合并训练集](../reports/experiments/2026-10-12-official-raw-augmented-training.md)，以及本轮 EXT379 只读数组统计。
- [S1] GOA multi-species README：<https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/README>
- [S1a] GOA UniProt 多物种文件目录：<https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/>
- [S1b] GOA release manifest：<https://ftp.ebi.ac.uk/pub/databases/GO/goa/current_release_numbers.txt>
- [S2] GO Consortium annotation downloads：<https://geneontology.org/docs/download-go-annotations/>
- [S3] GO evidence code guide：<https://geneontology.org/docs/guide-go-evidence-codes/>
- [S4] GO GAF 2.2 format：<https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/>
- [S5] UniProt FTP LICENSE：<https://ftp.uniprot.org/pub/databases/uniprot/LICENSE>
- [S6] GO citation policy / license：<https://geneontology.org/docs/go-citation-policy/>
- [S7] UniProt current release README：<https://ftp.uniprot.org/pub/databases/uniprot/current_release/README>
- [S8] NCBI RefSeq about / production policy：<https://www.ncbi.nlm.nih.gov/refseq/about/>
- [S9] NCBI molecular data usage policy：<https://www.ncbi.nlm.nih.gov/home/about/policies/>
- [G1] DeepGOPlus README：<https://github.com/bio-ontology-research-group/deepgoplus/blob/master/README.md>
- [G2] DeepGO-SE README：<https://github.com/bio-ontology-research-group/deepgo2/blob/main/README.md>
- [G3] DeepGOZero README：<https://github.com/bio-ontology-research-group/deepgozero/blob/main/README.md>
- [G4] GO 官方迁移公告与 URL 映射：<https://github.com/geneontology/go-technical-announcements/issues/20>、<https://github.com/geneontology/pipeline-from-goa/blob/main/docs/annotations-mapping.md>
- [G5] GO 官方站点元数据与工具 README：<https://github.com/geneontology/go-site/blob/master/README.md>
- [G6] TALE README / 真实数据目录：<https://github.com/Shen-Lab/TALE/blob/master/README.md>、<https://github.com/Shen-Lab/TALE/tree/master/data>
- [G7] DeepGraphGO 数据 README / 分卷目录：<https://github.com/yourh/DeepGraphGO/blob/master/data/README.md>、<https://github.com/yourh/DeepGraphGO/tree/master/data>
