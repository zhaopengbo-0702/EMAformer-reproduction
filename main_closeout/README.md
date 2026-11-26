# 主实验288次收尾

本目录按用户2026-11-23最新范围，仅验收主实验288次；不启动训练、不运行534次扩展或比较基线。原冻结代码、manifest、extensions保持不变。

用独立入口从完整结果树严格核验并生成EMAformer列的Table1/5、release与best-validation比较、参考差值和备份状态：

```bash
python main_closeout/build_report.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output results \
  --reference outputs/paper_reference_results.json \
  --destination outputs/main_closeout_report
```

此命令不训练，仅读结果并创建新报告目录。目标已存在则拒绝覆盖。若只有185次完成，会生成partial报告，SD在6个唯一seed未齐前保持null，不能伪装成主表最终结果。EMAformer数据可与论文参考值比较；其他模型仅是印刷参考，不能写成本次独立实测或重跑排名。

全部结果下载到短本地路径后，在本地重新运行，并加入一个或多个完整结果目录。每个目录应包含 `dual/` 与 `runtime_snapshots/`：

```bash
python main_closeout/build_report.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /path/to/download1/results \
  --local-results /path/to/download1/results /path/to/download2/results \
  --reference outputs/paper_reference_results.json \
  --destination outputs/main_closeout_local_final
```

local-results明确声明是本地下载副本；不要在云上把同NFS目录传为local-results。脚本会逐个run重验签名、metric、双checkpoint及对应snapshot；union按runid+signature去重。同id不同signature或者封存artifacthash冲突会失败。它不能替代归档传输的整体hash核对；原tar/receipt与下载验证应一并保存。

输出：`strict_aggregate.json`、`Table5_EMAformer_per_horizon.csv`（48配置×两角色）、`Table1_EMAformer_horizon_average.csv`（12数据集×两角色）、`release_vs_best.csv`、`acceptance.json`及`report.md`。paper_reference、measured、difference分字段，population/sampleSD同时保留，因为论文未披露ddof；按每个seed先跨四H平均，再统计Table1的六seedSD，不能用四H波动代替seedSD。

只有288唯一run全部核验、两角色Table5每48配置齐6seed、Table1每12数据集齐6seed、以及本地下载副本288run完整验签，才能标记主实验收尾完成。报告仍明确“主实验复现，未覆盖全文”，不要求扩展/基线训练作为本次验收条件。
