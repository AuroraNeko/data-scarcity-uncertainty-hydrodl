[English](README.md) | [简体中文](README.zh-CN.md)

# 水文深度学习中的数据稀缺与区间可靠性

本文档对应论文：

**Data Scarcity in Deep Learning Streamflow Forecasting: Prediction Skill,
Interval Reliability, and Conformal Calibration**

本项目研究监督训练记录长度如何影响径流点预测能力与预测区间可靠性。核心实验采用严格受控的 CAMELS-US 数据稀缺梯度：只改变训练目标年份数，流域集合、训练记录终点、序列记忆长度、批量大小、训练轮数、校准期和测试期均保持不变。

修正后的结论比旧稿更谨慎，也更符合真实实验结果。训练记录从 15 年缩短到 1 年时，逐流域 NSE 中位数由 `0.666` 降至 `0.511`；原始 90% 区间覆盖率由 `0.821` 降至 `0.776`。短记录模型的区间并未变窄，反而更宽，但 Winkler 分数明显恶化，说明预测误差增长快于区间宽度的自适应。全局 CQR 将边际覆盖率稳定在约 0.90，但高流量条件覆盖仍偏低，预测流量分层只能部分改善这一问题。

代码与紧凑结果的各版本归档统一使用 Zenodo concept DOI：
[10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955)。

## 修正后的主要结果

下表所有实验均使用相同的 50 个流域、180 天序列长度、batch size 512、固定 30 轮最终检查点、1995--2000 校准目标和 2000--2008 测试目标。结果为模型种子 42、123、456 的均值 ± 样本标准差。

| 训练目标记录 | 有效标签数 | 逐流域 NSE 中位数 | 原始 PICP | CQR PICP | 原始 MPIW | 原始 Winkler |
|---|---:|---:|---:|---:|---:|---:|
| 1 年 | 18,250 | 0.511 ± 0.006 | 0.776 ± 0.010 | 0.896 ± 0.002 | 0.442 | 1.137 |
| 3 年 | 54,750 | 0.570 ± 0.014 | 0.806 ± 0.013 | 0.906 ± 0.007 | 0.388 | 0.885 |
| 5 年 | 91,300 | 0.612 ± 0.023 | 0.815 ± 0.014 | 0.903 ± 0.005 | 0.376 | 0.808 |
| 15 年 | 266,322 | 0.666 ± 0.005 | 0.821 ± 0.009 | 0.901 ± 0.007 | 0.358 | 0.709 |

15 年减 1 年的逐流域 NSE 配对中位效应为 `0.140`，流域 bootstrap 95% 区间为 `[0.104, 0.173]`；94% 的流域在长记录条件下表现更好。原始 PICP 的端点差为 4.6 个百分点，95% 区间为 `[1.5, 7.7]` 个百分点。

定向稳健性实验保留了点预测结论，也揭示了区间响应的边界。综合五个互斥的 50 流域子集，逐流域 NSE 配对效应为 `0.154`，两层 bootstrap 95% 区间为 `[0.137, 0.170]`，250 个流域中有 96.8% 为正。综合原始 PICP 效应为 3.72 个百分点（`[2.11, 5.32]`），但只有 68% 的流域为正，且一个子集的区间跨越零。把两个端点都限制为 1,080 次优化器更新后，NSE 效应仍为正，为 `0.047`（`[0.022, 0.090]`），但 15 年原始区间转为过覆盖。含 83,523 个参数的标准 LSTM 得到 `0.134` 的 NSE 效应（`[0.095, 0.187]`）。因此，记录长度对点预测的影响具有可重复性，而区间响应对流域组成和训练预算更加敏感。

1 年条件下，按各流域观测流量三分位划分后的平均覆盖率为：

| 流量组 | 原始 QR | 全局 CQR | 预测流量分层 CQR | 观测流量分层参考 |
|---|---:|---:|---:|---:|
| 低 | 0.783 | 0.932 | 0.896 | 0.865 |
| 中 | 0.797 | 0.910 | 0.917 | 0.886 |
| 高 | 0.749 | 0.841 | 0.867 | 0.899 |

观测流量分层参考需要使用未知的测试观测来分组，只能用于诊断；预测流量分层使用校准期预测中位数阈值，在本研究的已见流域上可部署。

![修正后的数据稀缺效应](results/figures/corrected/fig1_scarcity_effects.png)

![修正后的条件覆盖结果](results/figures/corrected/fig2_conditional_coverage.png)

![多子集端点复现结果](results/figures/corrected/figS5_subset_replication.png)

![高流量覆盖压力测试](results/figures/corrected/figS3_high_flow_thresholds.png)

## 扩展校准与方法分析

除修正后的核心流水线外，仓库还包含论文补充材料（S13--S19）的扩展分析。
这些分析基于已保存的预测与紧凑结果表离线运行，脚本位于
[`revision_work/`](revision_work/)：

- **校准端变体**（`tier1b_*.py`）：序贯滚动/扩展窗口再校准、自适应保形推断
  （ACI）、上下侧非对称半径、干旱度四分位分组半径、校准窗口敏感性
  （1--5 个水文年，`tables/scarcity_windows/`），以及逐成员校准的 EnCQR
  对比（`tables/corrected_full_671/encqr_comparison.json`）。
- **局部化诊断**（`tier1b_*.py`）：逐流域校准池、三分位对十分位预测流量
  分层、按水文状态排列的残余异质性，以及可部署的中位数偏差校正实验。
- **扩展方法套件**（`tier3_*.py`）：五分量混合密度网络与 UMAL 式连续分位
  模型头、加噪声头与重选检查点的 MC Dropout 重训、50--99% 多名义水平分位
  可靠性，以及离线 CRPS 对比（`tables/tier3_671/`）。
- **空间迁移**（`tier2_ungauged*.py`）：冻结模型在 50 流域流水线从未使用的
  621 个流域上的端点评估（`tables/ungauged_scarcity/ungauged_summary.json`；
  逐种子中间产物属复现性产物，不入库）。

这些分析共同表明：所测试的校准端变体都不能消除区间质量上的稀缺梯度；
一阶中位数偏差校正不能弥合高流量条件覆盖缺口；该缺口是条件离散度问题
而非边际分布问题。

## 安装

```bash
git clone https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl.git
cd data-scarcity-uncertainty-hydrodl
pip install -r requirements.txt
```

如需复现本文使用的 CUDA 12.8 精确包版本：

```bash
pip install -r requirements-reference-cu128.txt
```

参考实验使用 Python 3.11、PyTorch 2.11.0+cu128 和一张 NVIDIA RTX 5060 Ti。训练和 50 次 MC Dropout 推理建议使用 GPU；已有结果表时，汇总与制图可在 CPU 上完成。

## 数据准备

```bash
python download_camels.py
python src/data/data_preprocessing.py
python experiments/analysis/audit_processed_data.py \
  --processed-dir data/processed/camels_us \
  --output results/tables/processed_data_audit.json
```

下载脚本使用 [Zenodo record 15529996](https://zenodo.org/records/15529996)，源数据 DOI 为 [10.5065/D6MW2F4D](https://doi.org/10.5065/D6MW2F4D)。

本项目组合使用 CAMELS-US v1.2 强迫/径流时序归档与 v2.0 属性集。共同处理日期为 `1980-01-01` 至 `2008-12-31`，终点由 Maurer 强迫数据决定，不再把 2009--2010 的缺失尾段静默填成标准化零值。

对每个强迫通道，线性插值使用 7 个连续值的上限，随后的向前和向后填补各使用 3 个连续值的上限。3 个流域缺少地质孔隙度，标准化后置零等价于相应训练流域均值填补；这 3 个流域均不在固定的 50 流域稀缺子集中。

每个处理文件包含 Daymet、Maurer 和 NLDAS 各 5 个命名列，共 15 个输入通道。上游归档中，Maurer 的 Tmax/Tmin 完全相同，NLDAS 的 Tmax/Tmin 也完全相同，因此实际包含 13 条不同的日序列。为兼容已训练检查点，本项目保留这些重复列，并由数据审计脚本明确报告。

## 复现修正后的流水线

可恢复流水线不会终止其他 Python 进程：

```bash
python experiments/orchestrator.py
```

只有当协议字段、检查点、预测文件及其 SHA-256 全部一致时，稀缺实验才会跳过。全流域评估脚本每次都会启动，并且只有在检查点、归一化统计、日期、序列长度、MC 次数和 batch size 的身份全部匹配时才复用分割缓存。点模型和种子 42 分位数模型还必须具有与检查点 epoch 和 SHA-256 一致的训练完成侧车，才会跳过训练。

已有全流域检查点时：

```bash
python experiments/orchestrator.py --skip-full-training
```

如已有紧凑稳健性结果快照，不重新训练 24 个多子集端点、6 个等更新端点和 6 个标准 LSTM 端点：

```bash
python experiments/orchestrator.py --skip-full-training --skip-robustness-training
```

也可以显式运行 12 个受控稀缺实验：

```powershell
foreach ($years in 1,3,5,15) {
  foreach ($seed in 42,123,456) {
    python experiments/scarce/train_scarcity_corrected.py `
      --years $years --basin-seed 42 --model-seed $seed `
      --seq-len 180 --batch-size 512 --epochs 30
  }
}
python experiments/analysis/analyze_corrected_scarcity.py
```

运行基于已保存预测的诊断和定向端点敏感性实验：

```bash
python experiments/analysis/analyze_conditional_robustness.py
python experiments/analysis/make_scarcity_replication_subsets.py
python experiments/run_scarcity_robustness.py --stage all
python experiments/analysis/sanitize_scarcity_checkpoint_metadata.py
python experiments/analysis/analyze_scarcity_robustness_experiments.py
```

稳健性运行器只会跳过协议字段、检查点、预测文件及其 SHA-256 全部匹配的结果；不完整或不匹配的产物会被明确报告，必须显式使用 `--force` 才会覆盖。

独立的 671 流域方法比较使用完整统一的校准/测试目标日期网格；缺失的观测径流始终由 mask 排除，不进入指标。点模型检查点按验证集 MSE 选择，分位数检查点则在 30 轮预算内按训练 pinball loss 选择，因此这里是统一评价网格上的诊断比较，而不是训练算法之间的因果比较：

```bash
python experiments/uncertainty/evaluate_corrected_methods.py \
  --batch-size 1024 --n-mc 50 --bootstrap 1000
```

50 流域子集的代表性诊断使用与该子集互斥的其余 621 个流域作为比较组：

```bash
python experiments/analysis/basin_representativeness.py
```

生成图表、导出论文 LaTeX 数值片段并检查一致性：

```bash
python experiments/analysis/make_corrected_figures.py
python experiments/analysis/export_manuscript_results.py  # 作者工作区
python experiments/analysis/verify_manuscript.py
```

## 仓库结构与公开策略

[`experiments/README.md`](experiments/README.md) 明确区分当前论文入口与仅用于开发历史、结果溯源的脚本；只有当前入口可以写入权威的 `corrected_*` 结果目录。

```text
.
|-- src/                              模型、损失函数和修正后的数据加载器
|-- experiments/
|   |-- scarce/train_scarcity_corrected.py
|   |-- uncertainty/evaluate_corrected_methods.py
|   `-- analysis/                     数据审计、汇总、图表和验证
|-- revision_work/                    补充材料扩展分析（S13--S19）
|-- results/
|   |-- tables/corrected_scarcity/
|   |-- tables/corrected_full_671/
|   |-- tables/full_checkpoint_training/
|   |-- tables/scarcity_replications/
|   |-- tables/scarcity_equal_updates/
|   |-- tables/scarcity_standard_lstm/
|   |-- tables/scarcity_windows/      再校准窗口敏感性
|   |-- tables/tier3_671/             稠密头、重训、可靠性、CRPS
|   |-- tables/ungauged_scarcity/     621 未见流域空间迁移
|   |-- figures/corrected/
|-- REPRODUCIBILITY.md
|-- requirements-reference-cu128.txt
`-- requirements.txt
```

原始/处理数据、模型检查点和逐样本预测数组不进入普通 Git 历史；紧凑 JSON/CSV 结果表和生成图表适合提交。正式 tagged release 可在存储允许时附带检查点和预测数组，结果元数据中的 SHA-256 可用于核对归档文件。

## 模型说明

LPU-Stream 使用 128 单元 LSTM，并通过静态 MLP 将 13 个流域属性嵌入 32 维空间。修正后的稀缺实验模型用 pinball loss 预测 0.05、0.50 和 0.95 分位数，并采用“中位数 + softplus 非负偏移”的单调参数化，因此分位数交叉率按构造为零。分位数模型含 104,099 个可训练参数。CQR 是训练后的校准步骤，不会重新训练网络。

## 复现与引用

详细协议、产物边界和审计命令见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。引用时应使用与实际代码版本对应的归档；整个发布序列的 concept DOI 为 [10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955)。

## 许可证

MIT，见 [LICENSE](LICENSE)。
