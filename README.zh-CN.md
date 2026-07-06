[English](README.md) | [简体中文](README.zh-CN.md)

# 数据稀缺与水文深度学习中的不确定性

本研究 **《How Data Scarcity Compromises Uncertainty Estimates in Hydrological
Deep Learning — and How to Fix It with Conformalized Quantile Regression》**
的配套代码。

本项目研究**训练数据量如何影响不确定性估计的可靠性**（而不仅仅是点预测精度）。我们在
CAMELS-US 数据集（671 个流域，三套气象强迫产品）上发现：将训练数据从 15 年缩减到 1 年，
90% 预测区间的覆盖率（PICP）从 83% 跌到 67%——远大于同期点预测精度 NSE 的下降
（0.884 → 0.719）。我们诊断了退化机理（区间系统性偏窄、校准曲线变平、流量区间不对称），
并证明 **共形分位回归（CQR）** 仅需一次前向传播、零额外训练成本，即可在所有数据量级别
将覆盖率恢复到约 0.87。我们进一步发现全局 CQR 修复的是**边际覆盖**而非高流量**条件覆盖**，
而分区间条件 CQR 可以弥补这一缺口。

> **状态：** 水文深度学习中数据稀缺与不确定性研究的代码。仓库包含示例结果 JSON 与图表；
> 24 GB 数据集与模型权重由下列脚本重新生成，不在仓库中。

---

## 核心结果

| 训练数据 | NSE | PICP（未校准） | PICP（CQR） | $q_{\text{cal}}$ | MPIW |
|---|---|---|---|---|---|
| 1 年  | 0.719 | 0.671 | **0.883** | 0.341 | 1.298 |
| 3 年  | 0.813 | 0.741 | **0.869** | 0.155 | 0.968 |
| 5 年  | 0.843 | 0.774 | **0.866** | 0.096 | 0.833 |
| 15 年（匹配 50 流域） | 0.884 | 0.828 | **0.876** | 0.042 | 0.662 |

数据稀缺下，**不确定性的退化远比点预测严重**，而 CQR 能稳健地恢复近目标覆盖率。
四行均使用相同的 50 个流域（seed 匹配），构成干净的组内数据量梯度。

### 边际覆盖 vs. 条件覆盖

| 流量区间 | 未校准 | 全局 CQR | 分区间 CQR |
|---|---|---|---|
| 低流量 | 0.91 | 0.996 | 0.885 |
| 中流量 | 0.56 | 0.938 | 0.876 |
| **高流量（洪水）** | **0.50** | **0.736** | **0.888** |

全局 CQR 恢复了边际覆盖率，但高流量（洪水）区间仍然欠覆盖（0.736 < 0.90）。分区间条件
CQR 将所有区间恢复到约 0.88。

<p align="center">
  <img src="results/figures/fig1_degradation.png" width="90%"><br>
  <em>点预测精度（NSE）下降温和，而区间覆盖率（PICP）骤降；CQR 可将其恢复。</em>
</p>
<p align="center">
  <img src="results/figures/fig2_method_comparison.png" width="90%"><br>
  <em>MC Dropout、Deep Ensembles、CQR 三者中，只有 CQR 以最窄区间、最优 Winkler 分数达到近目标覆盖。</em>
</p>

---

## 安装

```bash
git clone https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl.git
cd data-scarcity-uncertainty-hydrodl
pip install -r requirements.txt
```

训练推荐使用 CUDA GPU（实验使用单块 NVIDIA RTX 5060 Ti）；分析脚本亦可在 CPU 上运行。
测试环境：Python 3.11、PyTorch 2.x。

## 数据准备（CAMELS-US，约 24 GB）

```bash
# 1. 从 Zenodo 下载原始数据（约 12 GB zip + 属性文件）
python download_camels.py

# 2. 将 zip 解压到 data/raw/camels_us/，再预处理为逐流域 CSV
python src/data/data_preprocessing.py
```

将生成 `data/processed/camels_us/<basin_id>.csv`（671 流域，三套强迫产品的 15 个动态特征）
及 `data/metadata/` 下的归一化统计量。数据集**不**纳入 git（见 `.gitignore`）。

## 仓库结构

```
.
├── download_camels.py            # 从 Zenodo 获取 CAMELS-US
├── src/
│   ├── utils.py                  # 公共工具：get_device()、set_seed()
│   ├── data/                     # data_preprocessing.py（三套强迫）、dataset.py、compute_perbasin_stats.py
│   ├── models/                   # lstm、ea_lstm、tcn、transformer、lpu_stream
│   └── losses/                   # pinball_loss、physics_loss、cqr（校准器 + 指标）
├── experiments/
│   ├── baseline/                 # train_model.py（DL 基线）、train_xgboost.py
│   ├── scarce/                   # 数据稀缺实验（1/3/5/15 年，50 流域）
│   ├── uncertainty/              # 分位+CQR、MC Dropout、Deep Ensembles、公平评估
│   ├── physics_guided/           # 物理约束训练（辅助）
│   ├── analysis/                 # 制图、验证、诊断、跨区域、稳定性
│   └── orchestrator.py           # 全流程驱动（可恢复，带校验门）
├── paper/                        # manuscript.tex + 图表（HESS / Copernicus 格式）
└── results/
    ├── tables/                   # 结果 JSON（已提交，经 verify_manuscript.py 校验）
    └── figures/                  # 生成的图表 PNG + PDF（已提交）
```

## 复现实验

### 全流程（自动化）

```bash
python experiments/orchestrator.py
```

依次执行所有阶段——预处理（已缓存则跳过）、基线训练、分位+ensemble 训练、不确定性评估、
稀缺+跨区域实验、图表生成、论文数值校验——支持逐阶段恢复与超时保护。

### 分步执行

```bash
# 点预测基线（LSTM、EA-LSTM、TCN、Transformer、LPU-Stream）
python experiments/baseline/train_model.py --model lpu_stream
python experiments/baseline/train_xgboost.py          # XGBoost（单强迫 lag 特征）

# 逐流域 NSE 评估（CAMELS 标准指标 + bootstrap 置信区间）
python experiments/analysis/eval_point_perbasin.py

# 主不确定性结果：分位回归 + CQR 校准
python experiments/uncertainty/train_quantile.py                  # 单分位模型 + CQR
python experiments/uncertainty/retrain_ensembles_correct.py       # 5-member Deep Ensembles
python experiments/uncertainty/eval_fair_671.py                   # 公平 671 流域比较（CQR/MC/Ens）

# 数据稀缺实验（匹配 50 流域，1/3/5/15 年）
python experiments/scarce/train_data_scarce.py --years 1
python experiments/scarce/train_data_scarce.py --years 1 --no-static   # 消融

# 分析与制图
python experiments/analysis/diagnose_1yr.py             # 覆盖退化诊断 + CQR 分区间
python experiments/analysis/cross_region_validation.py   # 跨区域稳健性
python experiments/analysis/stability_1yr.py             # 多 seed 稳定性
python experiments/analysis/confidence_levels.py         # 90/95/99% 校准
python experiments/analysis/make_figures.py              # 从 JSON 重生成所有图表
python experiments/analysis/verify_manuscript.py         # 校验论文所有数值 vs JSON（101 项）
```

仓库中 `results/tables/*.json` 与 `results/figures/*` 为这些运行的示例输出。

## 模型 — LPU-Stream

一个轻量循环网络（103,969 参数）：15 个动态气象输入（五个变量 × 三套 CAMELS 强迫产品：
Daymet、Maurer、NLDAS）由 128 单元 LSTM 处理；13 个静态流域属性经 MLP（64→32）嵌入为 32 维
向量，拼接到每个时间步。线性头预测三个分位（0.05 / 0.50 / 0.95），用 pinball loss 训练。
CQR 在留出的 5 年校准期上做后处理校准。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
