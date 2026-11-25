# EMAformer 串行复现执行器

本目录不修改 `EMAformer/` 官方检出。代码哈希、运行配置和机器结果保存在输出目录；每次运行使用按哈希冻结的源码副本。没有任何云任务创建或多卡申请代码。

## 范围

- `manifest_released.json`：逐字解析官方脚本，支持嵌套 `for`，不执行 shell。官方 active 命令为 main 52、boost 48、lookback 38。Exchange 的 4 个脚本是论文之外的额外数据集，归入 `upstream_extra_exchange`。论文 main 48 配置各 6 seeds，其他脚本单 seed，共 378 次。
- `manifest_paper_owned.json`：论文作者方法及其变体，共 822 次训练。包括 main 48×6、7种新增消融×48、迁移108、MLP32、MeanInput16、另一个周期32、另5种历史长度×2数据集。比较基线独立重跑与checkpoint分析不在此训练清单。
- 六个 seeds 为本次自主选定 `2023..2028`。论文披露六个但未给编号；其他研究重复次数未披露。本清单对非main取一个 seed 2023，不能冒称论文完整原协议。
- 构造变体的精确定义与论文歧义见 `reproduction_extensions/README.md` 与 `outputs/paper_experiment_inventory.md`。

## 生成与自检

```bash
python reproduction/manifest.py --repo EMAformer --output reproduction/manifest_released.json
python reproduction/full_manifest.py --repo EMAformer --output reproduction/manifest_paper_owned.json
python -m unittest discover -s reproduction -p 'test_*.py' -v
```

这9个测试检查嵌套循环与注释展开、拒绝未知shell语法、主矩阵与822清单计数、跨进程排他锁与child独立持锁、resume验证身份/数值/文件哈希、unequal batch流式指标数值及zero-target处理。未执行训练，不能代替GPU smoke。

## 执行

必须先在平台确认没有其他 pending/running 作业，只申请 **一个任务、一张GPU**。在该任务内运行以下命令。runner 使用 kernel 文件锁，同一个 `--output` 上禁止两个并发队列；训练子进程仅一个，CUDA检查要求正好一个可见设备。请所有启动使用同一个输出根目录，禁止启动第二云任务。

实际训练child还持有机器内共用 `/tmp/emaformer-single-training.lock`（Windows为temp目录）。即使队列被SIGKILL而child仍在跑，重新启动也会失败，不会同时训练。不同output的队列也使用该child lock。默认路径不得为了绕过锁而改掉；若使用多个执行容器，文件锁不能代替平台的单任务申请限制，仍须先在平台核对。

```bash
python -u reproduction/run_queue.py \
  --manifest reproduction/manifest_paper_owned.json \
  --repo EMAformer --data-root /your/persistent/dataset \
  --output /your/persistent/emaformer-results \
  --mode dual --evaluation streaming --groups multivariate_forecasting
```

执行全部作者方法训练矩阵：

```bash
python -u reproduction/run_queue.py \
  --manifest reproduction/manifest_paper_owned.json \
  --repo EMAformer --data-root /your/persistent/dataset \
  --output /your/persistent/emaformer-results \
  --mode dual --evaluation streaming \
  --groups multivariate_forecasting armor_ablation transfer_original transfer_inverted transfer_armor mlp_backbone mean_input alternative_period lookback_emaformer
```

可用 `--seeds 2023`、`--ids RUN_ID`、`--max-runs 1` 选择正式配置的小范围检查；不降低epoch的检查属于实际完整训练，不是另造的smoke结果。默认遇错即停，查看对应 `train.log`。只有明确要继续后续独立配置才使用 `--continue-on-error`。

`--mode` 必选：

- `release` 保留官方 early-stopping 浅引用行为。实际返回权重可能是最后epoch，不能称严格best-validation。
- `best-checkpoint` 在内存保存点clone所有state tensor，确保最终恢复验证损失最优权重。与release分目录，严禁混合求均值。
- `dual` 一次训练保留官方release权重为primary，同时把真正best-validation快照另存CPU到 `checkpoint_best.pth`，对两套权重分别评估。primary在 `machine_metrics.json`，secondary在 `machine_metrics_best.json`，不需要重复训练。两者都通过完整性检查才complete。

默认 `--evaluation streaming` 逐batch执行同一官方测试loader、forward和切片；elementwise减法/平方保持float32，但采用float64全局求和再除全部元素数，避免不等长batch平均误差与巨量预测数组占用内存。其结果会与原版float32 NumPy归约有细微舍入差异。数值测试误差小于1e-7。`--evaluation original` 保留原版全量数组路径，适合小数据作逐项对照；大Traffic可能RAM不足。streaming保存前16窗口sample和一张曲线，不生成原版每20batch一张的PDF，数学指标相同定义。MAPE/MSPE零分母导致非有限时写null，不伪装为0；主验收MSE/MAE/RMSE须有限。

主训练按seed优先排序，先完成48个seed2023配置，再2024等；不会在同一配置上先完成6seed才去下一个数据集。每epoch记录官方训练时长及CUDA峰值allocated/reserved到 `epoch_resources.json`；测试秒数使用CUDA synchronize并另行测量。不同显卡/环境不能与作者4090时间直接声称一致。

除checkpoint模式外，均保留官方L1 criterion、官方loader划分/预处理/phase定义。默认DataLoader worker数显式降为2以节约实验室CPU，完整解析配置保存在 `effective_args.json`；不改学习率、epoch、batch或模型规模。

## 证据与恢复

每个run的目录为 `MODE/RUN_ID/attempt-NNN/`，包括 `run.json`、`status.json`、`invocation.json`、`train.log`、`environment.json`、`effective_args.json`、`pip-freeze.txt`、`machine_metrics.json`、`checkpoints/`、最多16个测试窗口的 `prediction_sample.npz`。完整预测数组原版也未保存；本执行器不把sample冒称全量预测归档。

子进程exit0、MSE/MAE/RMSE有限非负、checkpoint非空，才标记complete。resume还须签名/metrics/checkpoint哈希一致；签名绑定选定数据文件SHA256、代码、instrumentation与模式。队列开始先检查所有选中数据文件存在且写 `dataset_inventory.json`，每次attempt保存 `dataset.json`；数据改动不能复用旧完成状态。失败或损坏会新建attempt，保留旧证据。SIGTERM/SIGINT会终止当前child并标记interrupted，不假装完成。

`runtime_snapshots/` 保存所用源码和instrumentation；`source_inventory.json` 保存源代码逐文件哈希；`queue_progress.json` 是队列进展，不是论文完成证据。

汇总命令仅读取通过完整性验证的实测结果（会核对checkpoint哈希，可能耗时）。dual模式分开release和best；只有n=6才标记主表某H的six_seed_complete；population/sample SD同时列出，因为论文未说明ddof。论文参考字段与measured字段分离：

```bash
python reproduction/summarize.py --manifest reproduction/manifest_paper_owned.json \
  --output /your/persistent/emaformer-results --mode dual \
  --reference outputs/paper_reference_results.json --report outputs/measured_summary.json
```

## 当前实现限制

`--evaluation original` 会累积全部prediction/target到RAM，Traffic长H需要很大主机内存，资源不足时失败并保留日志。原版测试PDF已在每次绘制后close防止figure泄漏；streaming路径只存一张sample。注意力熵、CoV、tSNE、资源图、对照基线需独立脚本/材料完成，822次作者训练本身也不等于全文交付完成。
