# EMAformer 主实验数据下载

| 资料 | 获取方式 |
| --- | --- |
| 288 次训练的完整精度指标、训练日志中的数值和资源记录 | [下载指标和训练记录](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-data-20261127) |
| 4 份 ETT 输入文件 | [下载 ETT 原始数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-data-20261127) |
| Weather 气象输入文件 | [下载气象数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-weather-data-20261127) |
| ECL 用电量输入文件 | [下载用电量数据](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-ecl-data-20261127) |
| 576 份预测样本 | 每份包含前 16 个测试窗口的预测值，压缩包约 1.48 GB。[下载预测样本](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-predictions-20261127) |
| 576 份模型权重 | 每次训练对应两份权重，共 52 个可单独解压的压缩包，约 87.39 GB。[模型权重发布页](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/emaformer-main288-weights-20261127)（上传和校验完成后开放） |

每个下载页面提供附件字节数和 SHA-256 校验值。预测样本和模型权重的压缩包内附索引，记录数据集、预测长度、随机种子和权重类型。训练中保存的预测样本来自前 16 个测试窗口，测试真值可用下方列出的原始输入数据按相同划分重建。

## 训练输入文件

本项目的 288 次训练使用 12 个数据集。论文作者提供的压缩包 `iTransformer_datasets.zip` 包含本实验使用的全部输入文件。作者提供谷歌网盘和百度网盘两个下载入口。[谷歌网盘](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) · [百度网盘](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr) · [作者的下载说明](https://github.com/PlanckChang/EMAformer/blob/b8f37c0c7d089e7263bbaeb681672a94f5afe233/README.md)

下载后先核对文件字节数和 SHA-256，再按下表将 12 个文件放到 `EMAformer/dataset/`。实验使用的压缩包为 171,600,194 字节，SHA-256 为 `01d7016eaa3eb571b77542612219d116dfc6a9f5399586dea1f98261fc4d80c4`。逐文件的字节数、SHA-256、采样和训练集划分已保存在仓库。[数据文件校验清单](../datasets/manifest.json) · [运行说明](../reproduction/README.md)

| 数据集 | 相对于 `EMAformer/dataset/` 的文件路径 | 来源和许可 |
| --- | --- | --- |
| ETTh1、ETTh2、ETTm1、ETTm2 | `ETT-small/ETTh1.csv`、`ETT-small/ETTh2.csv`、`ETT-small/ETTm1.csv`、`ETT-small/ETTm2.csv` | 四份文件已与原发布者逐字节哈希核对。原文件允许在保留署名和条款的前提下按 CC BY-ND 4.0 分享。数据由 ETDataset 项目发布，其作者请求引用 Zhou 等人的 Informer 论文。[四份原文件](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) · [原始许可](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE) · [引用信息](https://arxiv.org/abs/2012.07436) |
| ECL | `electricity/electricity.csv` | UCI 原始用电数据由 Artur Trindade 提供，采用 CC BY 4.0。LSTNet 将 2012 至 2014 年的记录转换为 321 通道的小时数据。抽查原始数据的前 12 小时，将每 15 分钟的数值向上取整后按小时求和，所得 3,852 个值均与小时文件一致。实验文件的 26,304 行测量值与 LSTNet 文件逐行相同，时间列使用基准时间标签。[UCI 原始数据与许可](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) · [LSTNet 数据预处理说明](https://github.com/laiguokun/multivariate-time-series-data#electricity-consumption) |
| Solar | `Solar/solar_AL.txt` | 上述作者包提供 137 个阿拉巴马州光伏站点的 10 分钟版。全部 52,560 行数值与 LSTNet 文件一致。LSTNet 记录了美国国家可再生能源实验室的原始来源。[LSTNet 数据来源](https://github.com/laiguokun/multivariate-time-series-data#solar-energy) |
| Traffic | `traffic/traffic.csv` | 上述作者包提供实验所用的 862 通道小时级版本。全部 17,544 行测量值与 LSTNet 文件一致，基准文件附加了时间列。LSTNet 追溯到加州交通局的道路占用率记录。[LSTNet 数据来源](https://github.com/laiguokun/multivariate-time-series-data#traffic-usage) · [Caltrans PeMS](https://pems.dot.ca.gov/) |
| Weather | `weather/weather.csv` | 该文件源于马克斯·普朗克生物地球化学研究所 Beutenberg 气象站的 2020 年数据。实验文件的 52,696 个时间戳及每行 21 项测量值已与研究所数据核对。基准文件调整了文本格式，并将二氧化碳一列从 `CO2 (ppm)` 改名为 `OT`。原始数据采用 CC BY 4.0。[研究所的许可说明](https://www.bgc-jena.mpg.de/wetter/weather_data.html) · [上半年原始记录](https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2020a.zip) · [下半年原始记录](https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2020b.zip) |
| PEMS03、PEMS04、PEMS07、PEMS08 | `PEMS/PEMS03.npz`、`PEMS/PEMS04.npz`、`PEMS/PEMS07.npz`、`PEMS/PEMS08.npz` | 上述作者包提供实验所用的四份 NPZ。STSGCN 项目另外发布了对应名称的 Parquet 版本，方便查阅来源。[STSGCN 数据说明](https://github.com/Davidham3/STSGCN) · [Caltrans PeMS](https://pems.dot.ca.gov/) |

论文正文及附录可从 arXiv 原地址下载。实际运行使用 Python 3.10.13、PyTorch 2.0.0 和 CUDA 11.7，依赖锁记录了 47 个包及哈希。请从各软件发布渠道按锁文件取得依赖。[论文原文](https://arxiv.org/abs/2511.08396) · [环境锁文件](../datasets/wheelhouse_linux/requirements-linux-cp310.lock) · [本次复现环境说明](../复现报告.md)

Solar、Traffic 和 PEMS03、PEMS04、PEMS07、PEMS08 的实验用文件请从上方作者包下载。该六份基准文件的转载许可尚未确认。

## 无需登录的备用下载

论文作者发布的压缩包提供与本实验校验清单一致的文件。以下备用下载提供相同的测量值。文件格式和 SHA-256 请分别按各自发布者的清单核对。[本实验文件校验清单](../datasets/manifest.json)

- **Solar** 下载后解压 GZIP，即可取得与实验输入全部 52,560 行、137 列数值一致的文本。基准包采用不同的小数文本格式。[LSTNet 下载](https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz)
- **Traffic** 下载后解压 GZIP，即可取得与实验输入全部 17,544 行、862 列数值一致的文本。基准包中的 CSV 附加了时间列。[LSTNet 下载](https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz)
- **PEMS03、PEMS04、PEMS07、PEMS08** 的备用文件由 STSGCN 作者以 Parquet 格式提供。已逐项核对四份文件中的全部 59,060,448 个数值，流量及可用的占用率、速度均与本实验的 NPZ 一致。EMAformer 训练读取流量通道。STSGCN 作者对这套打包方式采用 Apache 2.0，原测量值由加州交通局 PeMS 提供。[PEMS03 下载](https://modelscope.cn/datasets/Davidham3/pems-stsgcn/resolve/master/PEMS03/PEMS03_series.parquet) · [PEMS04 下载](https://modelscope.cn/datasets/Davidham3/pems-stsgcn/resolve/master/PEMS04/PEMS04_series.parquet) · [PEMS07 下载](https://modelscope.cn/datasets/Davidham3/pems-stsgcn/resolve/master/PEMS07/PEMS07_series.parquet) · [PEMS08 下载](https://modelscope.cn/datasets/Davidham3/pems-stsgcn/resolve/master/PEMS08/PEMS08_series.parquet) · [数据来源与许可](https://modelscope.cn/datasets/Davidham3/pems-stsgcn)
