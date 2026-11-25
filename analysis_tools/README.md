# 验证后的结果与checkpoint分析

本目录是新增独立分析层，不修改已经冻结/上传的训练代码，不启动训练、优化器或云任务。所有入口首先核验manifest/run.json/index/status一致，重算queue签名，核验metrics和checkpoint的SHA256。失败、运行中、pending与损坏artifact均不得进入实测汇总。

## CPU汇总：Table1/Table5、Fig3

```bash
python analysis_tools/aggregate.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /persistent/emaformer-results --mode dual \
  --reference outputs/paper_reference_results.json \
  --report outputs/verified_analysis.json
```

- 每组保留source/instrumentation/mode/checkpoint role/evaluation、数据hash及完整effective配置签名；release与best-validation不混合。
- Table5只有manifest声明的**6个不重复seed全部完成**才输出总体/样本SD；不完整组SD为null，不会把单seed SD=0报告成重复稳定性。
- Table1先在每seed内平均完整4H，再对完整6seed均值/SD。不能将4个H的离散程度当seed SD。
- Fig3必须同seed下8组合×4H完整；ETT/PEMS还需四个数据集均完整。full六seed均值不会和单seed消融混比。归一化使用`(max MAE - MAE)/(max-min)`；token_only若不是最差会明确标记，而不强放中心。
- 论文参考值仅独立存在`published_reference_not_measured`，不补填缺失实测单元。

2026-11-22对当前本地空结果目录运行：**0个已验证训练、822 pending、0个可绘制完整雷达组**。这不代表论文预测实验已复现。

## checkpoint逐一分析

先运行一次CPU嵌入导出，可不装完整绘图环境：

```bash
python analysis_tools/checkpoint.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /persistent/emaformer-results --mode dual \
  --run-id EXACT_MANIFEST_ID --checkpoint-role best-validation \
  --actions embeddings --device cpu \
  --analysis-dir /persistent/analysis/EXACT_MANIFEST_ID-best-embeddings
```

从已核验checkpoint安全读取state_dict，始终使用`weights_only=True,map_location=cpu`。通过`inspect.signature(torch.load)`判断是否显式支持mmap：支持时启用，不支持的云端torch2.0采用普通CPU加载；不捕获错误后回退、不移除weights_only。analysis.json记录实际load策略。默认checkpoint上限1536MiB；超过需显式增加，CPU线程2。joint仅导出channel0切片。保留checkpoint hash、训练签名、原数组shape，后续才能做tSNE。已直接读取所下载torch2.0.0官方wheel的torch/serialization.py，确认weights_only存在、mmap不存在。

熵Table2与有界forward耗时：

```bash
python analysis_tools/checkpoint.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /persistent/emaformer-results --mode dual \
  --run-id EXACT_MANIFEST_ID --checkpoint-role best-validation \
  --actions entropy --device cuda:0 --data-root /persistent/dataset \
  --max-samples 64 --batch-size 1 \
  --analysis-dir /persistent/analysis/EXACT_MANIFEST_ID-best-entropy
```

分析默认CPU；GPU只能cuda:0且必须正好一个平台可见设备，不重写CUDA_VISIBLE_DEVICES。分析与训练共用默认`/tmp/emaformer-single-training.lock`排他锁，所以必须在单任务中训练已停止后串行执行。若训练指定了自定义lock，分析也必须传同一`--training-lock`。不能为分析申请第二任务。

执行模型前核验并使用该run保存的runtime_snapshot源码及instrumentation，不导入后来修改的当前模型。严格load_state_dict。数据也须匹配训练hash。phase0先筛选测试窗口，再按最后层先head平均、逐行log2熵、跨行/样本平均。默认只取前64个合格窗口，最多512、batch最多4；保存选中index和总候选数。此为**有界checkpoint诊断**，因为原文未给H/seed/样本聚合，不能冒称精确Table2复现。若需要完整phase0分析，可显式多次分段实现，当前入口不静默遍历巨量数据。

`--actions forward-latency`只测一个batch、2warmup+5次no_grad前向，GPU同步。它不是训练epoch耗时、不是全test耗时，不能拿来替代Fig8。不要与开启attention的结果混比。

`entropy_table.py --manifest ... --output ... --mode dual --analyses ANALYSIS_DIR_1 ANALYSIS_DIR_2 ... --report outputs/entropy_table.json`可再核验多个分析receipt，并且仅在同seed/variant/role/phase_offset的完整4H都存在时输出Table2候选行。候选行明确标记是否覆盖所有phase0窗口；同H重复会拒绝。四H等权平均是本次公开选择，论文原H聚合未披露。

## tSNE

```bash
python analysis_tools/tsne.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /persistent/emaformer-results --mode dual \
  --analysis-dir /persistent/analysis/EXACT-best-embeddings \
  --destination /persistent/analysis/EXACT-best-tsne --seed 0
```

先再验证completion/checkpoint/嵌入hash，CPU2线程，确定性取前最多1024点，上限2048；PCA最多50维后tSNE，seed0、1000步、init=pca、learning_rate=auto、perplexity=min(30,n-1)。这些是冻结的本次选择，论文没有给，不能要求逐像素一致。输出坐标/源index与sklearn版本。当前只输出可复查坐标，不伪造尚未训练的embedding散点。

## Fig8：只聚合真实记录，绝不为了测量重训

```bash
python analysis_tools/resources.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output /persistent/emaformer-results --mode dual \
  --report outputs/verified_resources.json
```

只对full与token_only已完成checkpoint读取真实epoch_resources/environment/train.log与测试metrics，交叉核验epoch时间行。配对要求同seed/H/GPU/torch/cuda/batch/role/evaluation。记录allocated/reserved与train/test不同峰值，绝不混为同一指标。原queue completion只封存checkpoint/metrics hash，auxiliary日志未在训练时seal，分析明确保留此证据局限，不能包装成不可篡改来源。V100测量只代表本机，不等同原RTX4090图。

## 真实结果绘图

`plots.py`仅在完整6轴雷达组存在时画图；空结果生成状态JSON而不会画“模拟结果”。使用figure-style/kernel.py并遵循render后逐图目视检查：

```bash
python analysis_tools/plots.py --aggregate outputs/verified_analysis.json \
  --destination outputs/analysis_figures \
  --style-kernel /path/to/figure-style/kernel.py
```

## 验证边界

```bash
python -m unittest discover -s analysis_tools -p 'test_*.py' -v
```

7项有意义的合成fixture检查覆盖零结果不得填值、artifact损坏拒绝、缺seed不报SD、准确六seed/双checkpoint分离、重复seed与缺消融拒绝、effective配置漂移无法组成六seed组，以及torch2.0/支持mmap的安全loader签名适配（无weights_only或safe加载失败时拒绝，绝不不安全回退）。合成fixture仅在临时目录中，不是论文训练结果。当前没有真实已完成checkpoint，所以checkpoint/tSNE/熵功能只完成编译和协议检查，**未宣称完成真实checkpoint端到端验证**。
