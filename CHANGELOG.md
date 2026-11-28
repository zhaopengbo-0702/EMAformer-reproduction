# 更新记录

这里记录 EMAformer 复现仓库的主要变化。日期均为北京时间。

## 2026-11-28

- 发布图 1、图 7 的相关系数与变异系数复算结果，以及首个 ETTh1 实验的训练曲线、测试指标、CPU 重放和模型嵌入。补充包附中文说明与逐文件校验值。[补充分析数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-supplement-20261128)
- 截至当日，576 份模型权重已打包为 52 个可独立解压的 ZIP，正在上传。25 卷原始归档另行备份，访问方式见恢复说明。[权重下载说明](docs/data-availability.md) · [原始归档恢复说明](docs/archive-recovery.md)

## 2026-11-27

- 发布 288 次主实验的完整精度指标、训练数值日志、资源记录和四份 ETT 输入。另发布 Weather 和 ECL 输入文件。[主实验数据与 ETT](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-data-20261127) · [气象数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-weather-data-20261127) · [用电量数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-ecl-data-20261127)
- 发布 576 份预测样本，每份记录前 16 个测试窗口的预测值。新增按数据集、预测长度和种子查找模型权重的索引。[预测样本](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-predictions-20261127) · [权重索引](results-summary/main288_weight_download_index.csv)
- 将仓库的主要阅读说明改为中文，补充 12 个训练输入的获取说明。[中文数据下载说明](docs/data-availability.md)

## 2026-11-26

- 提交 288 次主实验的本地验收结果。25 卷原始归档共 5,727 个文件，已通过本地校验。[本地验收回执](results-summary/local-main288-acceptance.json)
- 发布表 1、表 5 的 EMAformer 结果、逐种子指标和两类权重的对照，并汇总尚未开展的分支实验。[复现报告](复现报告.md)

## 2026-11-24

- 完成 12 个数据集、4 种预测长度、6 个随机种子的 288 次主实验训练。每次训练保留公开代码对应权重和验证集最优权重。[复现报告](复现报告.md)
