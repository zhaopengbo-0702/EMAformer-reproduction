# EMAformer 论文复现

本仓库记录 EMAformer 在多变量时间序列预测上的主实验复现。共完成并验收 **288 次训练**，覆盖 12 个数据集、每个数据集 4 种预测长度、每组 6 个随机种子。每次训练保留两份模型权重，共 576 份。消融实验、模型迁移和独立基线重训暂缓。

## 查看复现结果

| 想看什么 | 在哪里查看 |
| --- | --- |
| 复现结果和原论文有多接近 | [复现报告](复现报告.md) |
| 各数据集的总体预测误差 | [表 1 · 按数据集汇总](results-summary/Table1_EMAformer_horizon_average.csv) |
| 各种预测长度的详细结果 | [表 5 · 按数据集和预测长度汇总](results-summary/Table5_EMAformer_per_horizon.csv) |
| 每次训练的具体数值 | [逐种子指标 · 576 条记录](results-summary/main288_per_seed_metrics.csv) |
| 同一次训练的两份权重表现 | [公开代码对应权重与验证集最优权重对照](results-summary/release_vs_best.csv) |

表格中的 MSE 是均方误差，MAE 是平均绝对误差，RMSE 是均方根误差。三个指标均为数值越小，预测越准。用原始指标重新汇总后，逐种子指标与表格均值一致，标准差及相关差值的最大绝对偏差为 3.47 × 10⁻¹⁸。另保留早期从云端汇总抄录的五位小数版本，已与最终汇总核对。[本地验收回执](results-summary/local-main288-acceptance.json) · [早期表 1 记录](results-summary/table1-release-observed.csv)

下载结果表后，常见列名包括 dataset（数据集）、seed（随机种子）、horizon 或 pred_len（预测长度）、checkpoint_role（权重类型）、measured_mean（本次复现均值）和 published_reference（论文参考值）。

## 下载原始实验归档

原始实验归档通过 GitHub 草稿版分批上传。全量归档有 25 卷、共约 94.59 GB，包含模型权重、运行日志、预测样本和实验配置。25 卷及其中 5,727 个文件均已通过本地 SHA-256 校验。登录仓库所有者账号后可在草稿中查看已上传的分片和备份索引。[归档草稿](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/untagged-cdc84517e84231fb16f5) · [下载与恢复说明](docs/archive-recovery.md)

## 重新运行实验

- 运行步骤、参数和单卡设置见[复现脚本说明](reproduction/README.md)。主实验使用[运行清单](reproduction/manifest_paper_owned.json)中的多变量预测组 `multivariate_forecasting`。
- 数据下载源、文件校验值和划分方式见[数据清单](datasets/manifest.json)。
- 本次实验使用 Linux x86_64、Python 3.10、PyTorch 2.0.0 和 CUDA 11.7。[依赖版本及校验清单](datasets/wheelhouse_linux/requirements-linux-cp310.lock)
- 原作者代码固定于提交 `b8f37c0c7d089e7263bbaeb681672a94f5afe233`。[原作者项目中文说明](EMAformer/README.md) · [MIT 许可原文](UPSTREAM-LICENSE) · [第三方许可说明](THIRD_PARTY_NOTICES.md)
