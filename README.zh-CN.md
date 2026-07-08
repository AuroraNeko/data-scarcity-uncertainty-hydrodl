[English](README.md) | [简体中文](README.zh-CN.md)

# 水文深度学习中的数据稀缺与不确定性估计

本文仓库对应论文：

**How Data Scarcity Compromises Uncertainty Estimates in Hydrological Deep
Learning and How Conformal Calibration Mitigates It**

本项目研究训练数据不足时，深度学习径流预测模型的不确定性估计会如何退化。实验基于
CAMELS-US 的 671 个流域、三个气象强迫产品，以及轻量级 LPU-Stream 网络。核心结论是：
数据稀缺对区间覆盖率的破坏明显快于对点预测精度的破坏，而 split conformalized
quantile regression（CQR）可以在不重新训练神经网络的前提下恢复接近目标水平的边际覆盖率。

仓库包含训练、评估、分析脚本，机器可读的 JSON/CSV 结果表，以及已生成的论文图件。
原始 CAMELS-US 数据和训练检查点体积较大，不纳入版本管理。

## 主要结果

### 匹配流域的数据稀缺梯度

四组实验均使用同一个 seed-42 抽样得到的 50 个流域。

| 训练数据量 | pooled median-quantile NSE | 原始 PICP | CQR PICP | `q_cal` | CQR MPIW |
|---|---:|---:|---:|---:|---:|
| 1 年 | 0.719 | 0.671 | 0.883 | 0.341 | 1.298 |
| 3 年 | 0.813 | 0.741 | 0.869 | 0.155 | 0.968 |
| 5 年 | 0.843 | 0.774 | 0.866 | 0.096 | 0.833 |
| 15 年 | 0.884 | 0.828 | 0.876 | 0.042 | 0.662 |

从 15 年训练数据降到 1 年时，未校准区间覆盖率下降约 16 个百分点，而 NSE 的相对下降约
19%。这说明不确定性可靠性不能只作为点预测精度的附属指标，而应作为独立的评价对象。

### 边际校准与条件校准

1 年数据稀缺模型上的流量分区覆盖率如下：

| 流量区间 | 原始 QR | Global CQR | Predicted-regime CQR | Observed-regime CQR |
|---|---:|---:|---:|---:|
| 低流量 | 0.910 | 0.996 | 0.991 | 0.885 |
| 正常流量 | 0.560 | 0.938 | 0.925 | 0.876 |
| 高流量 | 0.504 | 0.736 | 0.791 | 0.888 |

Global CQR 能恢复边际覆盖率，但高流量事件仍然覆盖不足。Predicted-regime CQR 是可部署的
条件校准版本；Observed-regime CQR 使用真实测试期流量分区，因此作为诊断性上界。

<p align="center">
  <img src="results/figures/fig1_degradation.png" width="90%"><br>
  <em>数据稀缺对不确定性覆盖率的破坏明显快于对点预测精度的破坏。</em>
</p>

<p align="center">
  <img src="results/figures/fig2_method_comparison.png" width="90%"><br>
  <em>671 个流域上的 MC Dropout、Deep Ensembles、Deep Ensembles + CQR 与单模型 CQR 公平比较。</em>
</p>

## 安装

```bash
git clone https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl.git
cd data-scarcity-uncertainty-hydrodl
pip install -r requirements.txt
```

训练建议使用 CUDA GPU。只要结果文件已经存在，图件生成和论文数字核验可以在 CPU 上运行。
实验环境为 Python 3.11 与 PyTorch 2.x。

## 数据准备

下载并预处理 CAMELS-US：

```bash
python download_camels.py
python src/data/data_preprocessing.py
```

预处理后会生成：

```text
data/processed/camels_us/<basin_id>.csv
data/metadata/normalization_stats.json
data/metadata/basin_metadata.csv
```

每个流域文件包含 15 个动态变量（Daymet、Maurer、NLDAS 各 5 个变量）、13 个静态流域属性、
mm/day 单位的径流、缺测掩码，以及基于训练期统计量归一化后的特征。

## 仓库结构

```text
.
|-- download_camels.py
|-- configs/
|   `-- data_config.yaml
|-- src/
|   |-- data/
|   |   |-- data_preprocessing.py
|   |   |-- dataset.py
|   |   `-- compute_perbasin_stats.py
|   |-- losses/
|   |-- models/
|   `-- utils.py
|-- experiments/
|   |-- baseline/
|   |-- scarce/
|   |-- uncertainty/
|   |-- physics_guided/
|   |-- analysis/
|   `-- orchestrator.py
|-- results/
|   |-- tables/
|   `-- figures/
|-- REPRODUCIBILITY.md
`-- requirements.txt
```

论文源文件和最终投稿 PDF 属于发布包或投稿材料，不作为默认公开代码仓库内容。

## 复现实验

运行完整流水线：

```bash
python experiments/orchestrator.py
```

也可以按阶段运行：

```bash
python experiments/baseline/train_model.py --model lpu_stream
python experiments/baseline/train_xgboost.py
python experiments/analysis/eval_point_perbasin.py

python experiments/uncertainty/train_quantile.py
python experiments/uncertainty/retrain_ensembles_correct.py
python experiments/uncertainty/eval_fair_671.py

python experiments/scarce/train_data_scarce.py --years 1
python experiments/scarce/train_data_scarce.py --years 3
python experiments/scarce/train_data_scarce.py --years 5
python experiments/scarce/train_data_scarce.py --years 15

python experiments/analysis/diagnose_1yr.py
python experiments/analysis/cross_region_validation.py
python experiments/analysis/stability_1yr.py
python experiments/analysis/confidence_levels.py
python experiments/analysis/enhanced_robustness.py
python experiments/analysis/basin_representativeness.py
python experiments/analysis/make_figures.py
python experiments/analysis/verify_manuscript.py
```

如果 `paper/manuscript.tex` 存在，`verify_manuscript.py` 会同时核验论文文本中的数字；
在仅包含代码的仓库中，它会继续核验 JSON 结果，并跳过文本存在性检查。

## 模型简介

LPU-Stream 使用 128 单元 LSTM 编码动态气象序列，并用静态流域 MLP 将 13 个流域属性映射到
32 维嵌入。分位数模型预测 0.05、0.50、0.95 三个分位数，使用 pinball loss 训练，共
104,099 个可训练参数。点预测版本为 103,969 个参数。CQR 在保留校准期上进行后处理，不重新训练网络。

## 复现说明

时间划分、随机种子、校准设置、图件再生成和推荐发布文件清单见
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 许可证

MIT。见 [LICENSE](LICENSE)。
