# EMAformer 实验归档与恢复

主实验共 288 次训练。公开资料可以直接在仓库查看，大文件的下载和校验步骤见下文。[仓库首页](../README.md) · [复现报告](../复现报告.md) · [逐种子指标](../results-summary/main288_per_seed_metrics.csv) · [表 1](../results-summary/Table1_EMAformer_horizon_average.csv) · [表 5](../results-summary/Table5_EMAformer_per_horizon.csv)

**原始归档的访问方式**。登录仓库所有者账号后，可查看草稿归档和已上传的附件。附件的完整性以备份索引及校验记录为准。[归档草稿](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/untagged-cdc84517e84231fb16f5)

## 归档结构

主导出清单包含 `main288_001.tar` 至 `main288_025.tar`，共 25 个独立、未压缩 TAR，原始字节合计 94,589,265,920 字节（约 88.09 GiB）。清单描述源码与配置、数据、离线依赖、主实验指标、两套模型权重、预测样本及运行证据；不包含已安装的运行环境目录。相同内容的本机补充材料以独立补充包和 SHA 索引关联，避免重复保存。

GitHub 归档页的单个附件须小于 2 GiB，因此每个 TAR 可以再按原始字节顺序切成多个 `.part001`、`.part002` 等附件。这些是二进制片段，不是各自可解包的 TAR。默认片段上限为 1,900,000,000 字节；实际恢复以所选索引记录的边界为准。不要混用不同分片规格或不同版本索引。

`export_manifest.json` 的固定 SHA-256 为：

```text
fd048a8b7e4559ab847210e36e9af55250824aa99cd1fb3869f7f3cff84c7d0b
```

附件中清单可能命名为 `export_manifest_<hash前缀>.json`；只有文件内容哈希与上述值相同，才是本次主导出清单。`backup_index_<hash前缀>.json` 描述片段与远端资产映射，`release_parts_index.json` 或范围索引描述卷内顺序、偏移、字节数和 SHA-256。`complete_uploaded_volumes` 仅表示该份索引确认的卷，不能从文件名或上传时间推定所有卷完整。

## 从 GitHub 验证分卷

登录仓库所有者的 GitHub 账号，打开归档草稿，在附件列表中找到指向本仓库的备份索引。下面命令中的索引展示如何校验已上传的分卷。历史索引 `backup_index_8cb5a628ae19a770.json` 记录迁移前的仓库。[归档草稿](https://github.com/zhaopengbo-0702/EMAformer-reproduction/releases/tag/untagged-cdc84517e84231fb16f5)

从仓库根目录运行以下命令，将索引文件名换成当前草稿附件列表中的文件名。每卷需通过全部分片及整卷校验。校验模式只读取归档数据。需要保存恢复后的 TAR 时，将 `--verify-only` 换成 `--destination restored-archives`。

    python -m pip install requests
    python tools/archive/restore_release_archive.py --index-asset backup_index_7927a2e913664ac2.json --volume main288_001.tar --verify-only

仓库主分支的源码、配置、统计表与复现报告可公开浏览。归档暂存在草稿，截图、部署记录和第三方原文件随归档供仓库所有者恢复。

## 恢复与校验

1. 从有权限访问的附件中取得原始导出清单、与本次恢复对应的备份索引及所有片段。先核对清单整体 SHA-256，再核对备份索引中绑定的 `manifest_sha256`；对索引自身使用随附回执或独立可信记录中的 SHA-256。缺片、清单不符或状态不完整时停止，不补零或跳过。
2. 按索引 `volume_order` 查找每卷，在 `local_split_index.volumes` 中匹配同名记录。只接受 `status=complete` 的卷；按该卷 `parts` 数组顺序逐片处理。核对 `number` 连续、首片 `offset=0`、后续偏移等于此前字节数之和，再核对每片大小与 SHA-256。默认附件名与片段 `name` 相同；其他分片规格应使用备份索引 `remote_parts` 对应的实际附件名，不能按网页显示顺序猜测。
3. 将已验证片段以二进制模式依次写入新的临时 TAR，不插入换行，不解压单片。拼接后核对整卷字节数和 SHA-256，同时要求与片段索引及原始导出清单的同名卷一致；通过后才将临时文件改名为 `main288_XXX.tar`。保留原片段，直至后续验收完成。
4. 将恢复的 25 个 TAR 和清单放在同一目录，使用随附 `verify_volumes.py` 先验卷哈希，再逐成员验大小与哈希并安全解包。示例使用任意本地新目录，不依赖原部署路径：

```bash
python verify_volumes.py --manifest export_manifest.json --verify-only
python verify_volumes.py --manifest export_manifest.json \
  --destination restored --strip-emaformer-prefix
```

工具会回读解包文件并生成新的 `verification_*.json`，保留同名冲突文件并报错。单卷检查可加 `--volumes main288_001.tar`；这只验证所选卷，不能作为 25 卷齐全的证明。

5. 在 `restored` 项目根目录，用已有严格收尾工具核验原始指标、配置签名、源码快照与两套模型权重，并重新生成统计。`--local-results` 只能指向真实下载到本机的结果副本，目标报告目录须为新目录：

```bash
python main_closeout/build_report.py \
  --manifest reproduction/manifest_paper_owned.json \
  --output results --local-results results \
  --reference outputs/paper_reference_results.json \
  --destination outputs/local_acceptance
```

分片制作由 `release_split_local.py` 实现；上传工具的 `local-parts` 和 `tar-ranges` 模式采用相同原始字节边界。恢复端均按上述片段算法处理。单纯重新运行分片工具不能恢复丢失的源 TAR，也不能代替逐卷解包和科学验收。

补充包按其独立索引核对外包和每成员 SHA-256。索引中的云卷引用通过主清单 SHA-256 定位，包内别名引用已验证成员；按原相对路径恢复到单独目录，保留来源映射。不要把补充历史状态或截图当成最终验收。

## 结果范围与使用

主实验为 12 数据集 × 4 预测长度 × 6 种子，共 288 次训练；每次训练保存两份权重。`release` 对应原作者公开代码的保存方式，`best-validation` 对应验证集表现最佳时保存的权重。已发表基线数值是论文参考，不代表独立重训。消融、其他模型迁移等暂缓分支不因归档传输而变成已完成。

上游源码使用 MIT，保留原版权与第三方声明。数据、论文、依赖包和截图各自适用其来源条款；源码 MIT 不覆盖整份原始归档。公开数值摘要与原始材料的再分发范围应区分。本文不包含平台地址、账户凭据或认证链接。
