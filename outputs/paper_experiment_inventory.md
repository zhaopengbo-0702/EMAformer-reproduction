# EMAformer 全文实验核对清单

核对日期：2026-11-21。本文档为论文与代码证据清单，不代表实验已运行或完成。

## 1. 原始材料与复现范围

- 用户提供的论文：`EMAformer Enhancing Transformer through Embedding Armor for Time Series Forecasting.pdf`，9 页 AAAI-26 正式版，页码 28636-28644，SHA256 `a6f30788c18fab1fe83bafbefa063ee93d77d63c4fba9e40b7bb4a810fb9c56c`。
- 正式版第 1 页明确提供 Extended version 链接。已从 `https://arxiv.org/pdf/2511.08396` 获取扩展版，保存为 `paper/EMAformer_arxiv_extended.pdf`；内容为 arXiv:2511.08396v1，2025-11-11，14 页，SHA256 `8066963e80a2a97a2044a4c5636fe9a096ce47d784ce6d78e2ed690f24f50b9d`。
- 扩展版额外包含附录 A-F、完整 48 组预测步长结果与 6 随机种子说明、表示可视化、资源开销、历史窗口实验。因此“全文复现”按正式版加扩展版证据清单解释。
- 上游代码审阅版本 `b8f37c0c7d089e7263bbaeb681672a94f5afe233`。论文超参数缺失时，官方代码脚本与默认值构成“官方代码路线”，并须单独记录，不能称为论文已写明。
- 数值参考表已保存到 `outputs/paper_reference_results.json`，包含 Table 1-6；所有数值均标记为已发表参考值，不能写进本次实测结果。

## 2. 数据、预测任务与训练协议

输入历史长度 L=96；多变量预测所有通道；PEMS 预测长度为 12、24、48、96，其余数据集为 96、192、336、720；报告 MSE、MAE，越低越好。论文 §4.1 只称使用主流 train/valid/test 划分与 z-score，并未列出准确划分边界。优化器为 Adam，训练损失为 L1。作者机器为单张 GeForce RTX4090。不同 GPU 的时间和内存结果只可作本机测量，不能宣称精确重现 4090 资源数值。

| 数据集 | 通道数 | 论文 Table6 时间步 | 采样间隔 | 主实验周期 P | 日周期 | 周周期 |
|---|---:|---:|---|---:|---:|---:|
| ETTh1 | 7 | 14400 | 1h | 24 | 24 | 168 |
| ETTh2 | 7 | 14400 | 1h | 24 | 24 | 168 |
| ETTm1 | 7 | 57600 | 15min | 96 | 96 | 672 |
| ETTm2 | 7 | 57600 | 15min | 96 | 96 | 672 |
| ECL | 321 | 26304 | 1h | 168 | 24 | 168 |
| Solar | 137 | 52560 | 10min | 144 | 144 | 1008 |
| Traffic | 862 | 17544 | 1h | 168 | 24 | 168 |
| Weather | 21 | 52696 | 10min | 144 | 144 | 1008 |
| PEMS03 | 358 | 26208 | 5min | 288 | 288 | 2016 |
| PEMS04 | 307 | 16992 | 5min | 288 | 288 | 2016 |
| PEMS07 | 883 | 28224 | 5min | 288 | 288 | 2016 |
| PEMS08 | 170 | 17856 | 5min | 288 | 288 | 2016 |

Table6 的 ETT 行数不是常见原始 ETT CSV 总长度；官方 loader 使用 20 个 30 天月，ETTh 范围 14400、ETTm 范围 57600。数据核验应同时报告 raw 行数和实际使用行数。

官方 loader 的划分证据 `data_provider/data_loader.py`：ETTh train=8640、valid=2880、test=2880；ETTm 分别 ×4。Custom 使用 70% train、20% test，valid 为剩余；Solar 为 floor(70%)、floor(10%)、floor(20%)；PEMS 为 floor(60%)、floor(20%)、floor(20%)。valid/test 左侧允许 L 步训练/前段历史作为上下文，但预测目标必须在其所属区间。Scaler 只在训练区间拟合。PEMS 读取 `data[:,:,0]`。实际 floor 后边界应按代码保存，不能仅写近似比例。

扩展版 Table5（第14页，已渲染目视核验）明确报告 **6 个随机种子的均值与标准差**，但没有披露种子编号，也没有披露 SD 的 ddof。官方 `run.py` 固定 seed=2023，`--itr` 默认1；单纯执行原脚本一次不等于主表六次重复。复现应声明自主选择的6个种子，保存逐种子结果；主结果至少48×6=288次正式训练。不能把四个 horizon 的离散程度当作 seed SD。

## 3. 所有表图、主张与验收证据

| 项目 | 来源页 | 精确实验范围 | 必须保存/核验的结果 |
|---|---|---|---|
| Fig1 CoV | 正式p1；扩展p1/附录A p11 | ETTh2、ETTm2，每日 Pearson相关矩阵，再跨日求均值/总体SD与CoV | 两个7×7 CoV矩阵、每日相关矩阵、数据段与日分组规则 |
| Fig2 模型结构 | 正式p3；扩展p3 | token+C+P+J相加后Transformer与MLP输出 | 实现对照表、参数量、前向检查；无需额外训练 |
| Table1 主结果 | 正式/扩展p5 | 12数据集×4H；与9基线；每数据集跨H平均 | 48×6主模型逐种子MSE/MAE、12平均、排名、相对改进 |
| Fig3 三嵌入消融 | 正式/扩展p5 | 8组合×12数据集×4H；ETT/PEMS各组内平均 | 原始MAE，max-min归一化规则与雷达图，不能只存归一化图 |
| Table2 注意力熵 | 正式/扩展p6；扩展附录B p11 | token_only、token+C、full；ETT四组+ECL/Solar/Traffic/Weather；phase0 | 最后一层各head注意力、先head平均再逐行熵再平均；记录聚合样本/H/seed |
| Table3 向其他Transformer迁移 | 正式/扩展p6 | Reformer/Informer/Flowformer × original/inverted/embedding × ECL/Traffic/Weather ×4H | 108配置结果、27个H均值、相对inverted改进 |
| Fig4 MLP替换 | 正式/扩展p6 | ECL/Solar/Traffic/Weather/PEMS03/04/07/08 ×4H | 32 MLP配置；保持其他超参；与Full/CycleNet/TQNet比较MAE |
| Fig5 MeanInput | 正式/扩展p7 | ETTh2/ETTm2/Solar/Weather ×4H | 16配置；与DLinear/Crossformer/SCINet MAE；说明归一化保留信息 |
| Table4 日/周周期 | 正式/扩展p7 | 8个非PEMS数据集×2周期×4H | 64配置；主表已覆盖32，仅额外32；周期选择以valid判定更稳妥，勿用test调参 |
| Fig6 表示可视化 | 扩展p8（正文解释p7） | ECL/Solar的channel；Traffic的phase与joint channel0 | checkpoint、原始嵌入、tSNE种子/参数，原文未给tSNE超参与选哪个H |
| Fig7 CoV中间量 | 扩展p12 | Fig1两数据集的每日相关均值、SD | 可与Fig1共同计算，无需训练 |
| Fig8 资源开销 | 扩展p12；说明p13 | full vs ablated；全部12数据集×4H；ETT/PEMS再组内平均 | 每epoch训练秒、测试秒、GPU内存MB；warmup/sync/GPU型号/批量、allocated和reserved定义 |
| Fig9 历史窗口 | 扩展p12-13 | ECL/Traffic；L={48,96,192,336,528,720}, H=96；模型EMAformer,DLinear,PatchTST,iTransformer,CycleNet,TQNet | 72配置（与主表L96,H96重用12，新增60）与MSE曲线 |
| Table5 完整主表 | 扩展p14 | Table1主模型的48配置×6seed | 每H的均值SD及跨H平均；与Table1属于同一训练集合 |
| Table6 数据统计 | 扩展p13 | 12数据集 | 本地/云端文件SHA256、形状、缺失率、实际训练划分 |
| Algorithm1 | 扩展p13 | 模型伪代码 | 与正文及官方实现差异说明 |

8种消融组合为：Only Token，Token+C，Token+P，Token+J，Token+C+P，Token+C+J，Token+P+J，Full。Fig3 原文将 token_only 放在中心（最大MAE），越远越优；建议保存明确归一化公式与每轴上下界。

## 4. 训练数量与资源约束

以下先按“一个配置一个种子”计数，不承诺未知协议的精确种子设置。

| 部分 | 不重用前配置数 | 新增配置数（可复用主结果） |
|---|---:|---:|
| 主模型 |48|48|
| 8组合消融 |384|336|
| 其他Transformer迁移 |108|108|
| MLP版本 |32|32|
| MeanInput |16|16|
| 日/周周期 |64|32|
| 本方法历史窗口 |12|10|
| 注意力/表示/资源/CoV |checkpoint派生|0（若训练时已记录资源）|
| 合计，本方法及论文新变体 |—|582|

主模型必须6seed，其他部分论文未明确给repeat：因此“一次/配置的扩展全文覆盖，加主模型6seed”下限是 **582+48×5=822次训练**。若全部新实验一律6seed，为3492次。六seed扩展实验是本次选择，不能称论文明确要求。

9个Table1比较基线各48配置=432次/seed；Fig4和Fig5相关基线已包含其中。Fig9额外基线为5模型×2数据集×5个非96历史窗口=50次/seed。完整独立重跑全部比较基线总计再加482次/seed。全表图以1seed覆盖且main6seed时为1304次；全部6seed为6384次。该计数假设相同模型/数据/H/seed/超参可真正复用；如迁移脚本与主脚本不同超参，应按配置hash去重而非仅凭名字去重。

“论文表中引用原作者基线”与“本次独立重跑基线”必须在最终报告中分列，前者不计入本次完成训练数。配置总数不等于云任务数：所有训练应由一个串行队列依次执行，一次只能有一个平台任务，每个任务一张GPU；CPU准备步骤也先检查现存任务，不能额外申请第二任务。

## 5. 重要歧义、差异与不可暗中处理事项

1. **phase的off-by-one**：论文Eq6与Algorithm1使用最后已观察点t；官方loader取`cycle_index[s_end]`，而x是`[s_begin:s_end]`，故相位是首个未来点。两者不能无标记混用。官方代码路线保留原行为，论文索引路线如测试应单独标记。
2. **LayerNorm位置**：正文Eq9/10为post-LN，附录Algorithm1为pre-LN。官方EncoderLayer为post-LN，应记录论文内部不一致。
3. **额外残差与输出头**：官方EMAformer把encoder输出与原embedding再相加后输入3层MLP。正文只粗述MLP输出，Algorithm1没有该残差。采用官方实现需披露。
4. **MLP替换**：原文为去除Transformer、保留预测头且不调其他超参；未明确当enc_out+enc_origin残差存在时应保留哪一支。保留一份embedding直接入projector是可解释的复现选择，必须写明。
5. **MeanInput**：原文称把输入替换为其均值，经pre-normalization后近零，但未说明均值为全局还是每窗口，未说明训练/测试都替换还是仅推理替换，也未说明是否保留原始instance mean/std。Solar官方脚本use_norm0，不能默认所有数据都做实例归一化。可操作主路线为将standardized历史置零、use_norm1时保留原始实例均值和SD做denorm，并暴露window_mean替代；这是明确声明的复现选择，不是原文已确证协议。保留均值和SD时，输出仍受局部统计影响；不能声称完全不含历史信息。
6. **熵的log底**：附录公式只写log，但明确`log7≈2.81`，因此与表对应应使用log2。自然对数最大为1.946，无法比较。token-only高于log2(7)也应排查timestamp额外token、dropout、样本聚合，不可改值贴表。
7. **熵聚合细节**：phase0、最后层、先head平均再row熵已明；但具体test样本数量、选哪个H/seed以及如何跨H平均未明。
8. **CoV**：附录公式为sd/mean，population SD(1/N)明确；mean接近0时极不稳定。2026-11-21已独立复算，使用raw全部行、连续24点块（含尾块）、pandas Pearson、跨块跳过NaN、ddof0与abs(mean)后，ETTh2/ETTm2两张Fig1全部数值在两位小数精确匹配。ETTm2的24点仅6小时，与“每天”不符；真正96点日版本已另存。负均值/CoV实际存在，论文图展示绝对值。证据见`outputs/cov_analysis/verification.json`；精确匹配不代表已取得作者私有作图代码。
9. **资源图ablated定义**：caption只写Ablated model，正文称三种辅助embedding带来的开销，合理对照为token_only，但未显式列出配置；测量峰值allocated/reserved、是否forward/backward与optimizer state包含均未写。
10. **tSNE**：未提供perplexity、初始化、迭代数、随机种子、选择checkpoint/H。不能要求逐像素一致，只能在冻结设置下检验定性现象。
11. **误差条**：Table5印刷的PEMS03跨H平均MSE SD=0.007，高于每H四个印刷SD。如以同seed跨H均值解释，其值存在疑点。保留原值并报告，不擅自更正。
12. **主张排名**：20/24最佳、3/24第二意味着还有1例并非前二；正文一句“remaining3”没有覆盖全部剩余4例。根据Table1数值已复算：ETTm1 MSE第三；ETTh2/ETTm2/Traffic MSE第二；其余20项第一。
13. **相对改进**：2.73%/5.15%是相对每数据集最佳非本方法（本方法胜时即第二名）的均值，需明确负提升（本方法落败）保留，不能只平均获胜项。根据印刷Table1数值已复算MSE=2.726521%、MAE=5.149025%，与摘要四舍五入吻合。

## 6. 未披露超参数与完成判据

论文未完整披露：6seed编号、除主表外的重复数、优化器betas/eps/weight decay、学习率/调度、epoch/early stop、batch size、d_model/d_ff/heads/layers、dropout、tSNE设置、资源测量协议。可通过官方脚本与run.py默认值冻结复现，但每个run须保存完整解析后配置与代码hash。训练损失应以真实criterion检查：官方代码为L1Loss，不能因`--loss MSE`默认标签误写成MSE训练。

完成证据必须包括：下载数据与hash；所有计划配置的pending/running/succeeded/failed统计；每次逐seedMSE/MAE与checkpoint；主表/变体/分析图生成脚本；与论文参考值差异；未完成和缺失协议；完整结果本地备份。只生成计划、跑通smoke或提交云任务均不能标记“全文复现完成”。

