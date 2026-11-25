# 可追踪的复现模型扩展

这些文件位于官方 checkout 之外。它们不会修改论文作者源码，也没有启动训练或占用云 GPU。

## 接入

Python 搜索路径需要同时包含项目根目录和 `EMAformer` 官方代码目录。

```python
from reproduction_extensions import register_models

# 在 Exp_Long_Term_Forecast._build_model 调用 Model(config) 前运行：
register_models(self.model_dict)
# 扩展配置可由 wrapper 的 --extra-json 注入 self.args。
```

registry 仍然使用上游的 module.Model(config) 接口。覆盖 `EMAformer/iInformer/iReformer/iFlowformer`，新增 `EMAformerMLP/EMAformerMeanInput`，以及丢弃额外 phase 参数的 `OriginalTransformer/OriginalInformer/OriginalReformer/OriginalFlowformer` 适配器。后四者只是签名适配，没有变更原模型的预测。

## 配置规范

| 字段 | 默认 | 语义 |
|---|---|---|
| armor_variant | 不指定 | 可选 ablation 名称，决定三种嵌入默认开关 |
| armor_channel | 1 | channel embedding 0/1，可覆盖 variant |
| armor_phase | 1 | phase embedding 0/1，可覆盖 variant |
| armor_joint | 1 | joint embedding 0/1，可覆盖 variant |
| backbone | transformer | transformer 或 mlp |
| mean_input | 0 | 是否干预历史输入 |
| mean_input_strategy | zero | zero 或 window_mean，在上游实例归一化之后执行 |
| mean_input_apply | both | both/train/eval，控制训练、评估阶段 |
| phase_offset | 0 | 输入 phase 查表前偏移；0 保持官方代码，-1 对齐论文“最后观察步” |

`armor_variant` 的八种取值为 `token_only/channel/phase/joint/channel_phase/channel_joint/phase_joint/full`，对应 `(C,P,J)` 为 `000/100/010/001/110/101/011/111`。关闭的嵌入参数会从模型删除，不会被 optimizer 更新或虚增参数量。删除发生在原始初始化之后，所以启用部分的初始权重与相同 seed 的原模型一致。

`EMAformerMLP` 强制 backbone=mlp，`EMAformerMeanInput` 强制 mean_input=1；也可直接用 `EMAformer` 配置组合，不需要新名字。设置 model 为新名字时 wrapper 必须调用 register_models，即使不传其他扩展字段。

例子：

```json
{"armor_variant":"token_only"}
{"armor_channel":1,"armor_phase":0,"armor_joint":0}
{"backbone":"mlp"}
{"mean_input":1,"mean_input_strategy":"zero","mean_input_apply":"both"}
```

## 论文无法唯一确定的细节

- **MLP**：扩展论文 §4.3 表述为移除 Transformer、保留 prediction head、其他超参数不变。本实现删除整个 encoder，直接执行 `projector(tokens)`。它不使用 `Identity(tokens)+tokens`，以免把输入无依据地翻倍。官方 EMAformer 原本存在 encoder output + tokens 的跳连，但论文未明确移除 encoder 后该跳连如何处理；需在结果中标记为新增实现解释。
- **MeanInput**：正文为“replaced input series with its mean (effectively zero, due to pre-normalization)”。没有代码和足够细节确定全局/窗口均值、训练/评估阶段、归一化统计量。本实现 primary 是把标准化历史置零、训练与评估都使用，并保留上游实例归一化的原均值/标准差用于反归一化；Solar use_norm=0 则没有实例统计量。可选 window_mean 为保留输入在时间维的均值并广播。**保留实例统计量时不能声称消除了全部局部信息**。primary 与其他策略都必须在输出配置注明。
- **相位**：论文定义最后观察步，上游加载器却取第一预测步。默认 offset=0 忠实官方代码。offset=-1 只是敏感性分析选项，不能与默认结果混为一类。
- **骨干细节**：EMAformer 增强版保留官方三层 head+跳连，其余三个增强注意力模型保留官方单 Linear head 且无额外全局跳连；因此不能把它们当成“只换注意力”而隐藏结构差别。三个 i* 增强模型也保持上游无条件实例归一化行为。

## 分析接口

`analysis.py` 提供以下函数，均不训练：

- `parameter_overhead(model)` 返回实际总参数、训练参数、三种 armor、encoder、projector 各自参数量；关闭嵌入不计数。
- `export_embeddings(model, directory, channel=0)` 输出 `learned_embeddings.npz` 和对应 JSON。joint 只导出指定通道，避免复制整个大型 embedding；原始张量可用于有记录的 PCA/t-SNE 绘图。
- `AttentionEntropy(phase=0, base=2, phase_offset=0, cycle=None)`，每 batch 调用 `update(attns, phase)`，最后 `result()`。最后一层先对 head 求均值，再逐 row 算熵，最后平均 row 和 phase=0 样本；只维护标量统计。base2 来自附录 `log7≈2.81` 的数值，公式本身没有明确底数。
- `AttentionCapture(model, max_samples=4)` 上下文保存有限样本的最后一层 head 平均注意力，退出时自动移除 hook；调用 `export(path)` 保存 NPY。启用这些分析时需 output_attention=True 且 model.eval()；无法提供 attention 的 Reformer/Flowformer 不能伪造熵。
- `measure_inference(model, forward_args, warmup=2, repeats=5)` 返回 eval/no_grad 的 forward 中位耗时，GPU 时同步计时并记录 peak allocated bytes，CPU 时明确标记 cpu。无数据读写和训练，比较模型时须严格保持硬件、batch、shape、dtype、attention 配置一致。

## 验证

运行 `python reproduction_extensions/check_models.py`。本地使用 Python3.9、PyTorch2.7.0+cpu、隔离的纯 Python依赖，只做小张量前向/反向与语义检查，不是论文实验。

2026-11-21，十一项检查通过，结果见 `outputs/extension_model_checks.json`：四个默认扩展模型与官方模型初始 state_dict 及预测逐位一致；三种 embedding 均有有限非零梯度；八种消融均可运行且参数真实删除；MLP 不混合通道；norm0 置零历史输出不依赖历史；phase_offset 等价于显式相位移动；均匀与 one-hot 熵达到理论边界且正确筛选 phase0；注意力 capture 遵守样本上限；四个 temporal 原模型适配器都保持输出形状且不使用 dataset phase。

这些检查不证明云环境兼容、显存足够或论文数值已复现。正式运行必须记录实际环境并执行单卡 smoke。静态图片目录不属于新实验输出。
