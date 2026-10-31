[English](README.md) | [简体中文](README.zh-CN.md)

# 监督训练记录长度对径流预报点预测技巧与区间可靠性的影响

在保留独立校准记录的前提下，监督训练记录的长度对深度学习径流预报究竟有多重要？
本研究的干预是**只削减模型训练标签、校准观测单独保留**：一年模型仍需校准数据，
并不等同于只有一年全部径流观测的流域。

## 方法

- **模型**。所有构造共用同一 LSTM 骨干（LPU-Stream），输入 180 天气象驱动
  （记忆敏感性实验用 365 天）与静态流域属性，输出对数径流的单调
  0.05/0.50/0.95 分位数三元组；TCN、标准 LSTM 与 13 分位数头作为架构敏感性。
- **校准**。原始分位数预测用 split conformalized quantile regression（CQR）
  在任何模型都不参与训练的五年窗口（1995–2000）上重校准，并比较全局、
  逐流域与按预测流量分层三种变体。
- **稀缺性设计**。训练标签限制在截止 1995 年 9 月的嵌套 1/3/5/15 年窗口内；
  校准池与测试期（2000 年 10 月 – 2008 年 12 月）对所有训练长度固定，
  唯一变化的是监督记录长度。
- **规模**。50 个主流域 × 三个模型种子；四个互斥 50 流域子集复现端点
  （合计 250 个独立流域）；671 流域八种不确定性构造对比；以及 48 个补充
  端点拟合（替代训练年份、独立验证、多名义水平、卷积架构、匹配记忆长度）。
- **推断**。所有效应均附带流域级 bootstrap 区间，同时对流域与 HUC2
  聚类重采样。

## 关键结果

每条结论标注生成脚本：

| 结论 | 数值 | 生成脚本 |
|---|---|---|
| NSE 中位数 1年→15年 | 0.511→0.666;配对效应 0.140 [0.104, 0.173] | `coverage_groups.py` |
| 原始 90% 池化覆盖 1年→15年 | 0.776→0.821;等流域效应 +4.6pp [1.5, 7.7],72% 流域为正 | `coverage_groups.py` |
| 一年高**预测**流量覆盖 global→分层 CQR | 0.832→0.882;+5.0pp,流域/HUC2/同步年重采样下均正 | `coverage_groups.py` |
| 稠密分布评分(13点 IQS,log1p 尺度) | MDN 0.077511、高斯头 0.078314、噪声MC 0.081902;解析 CRPS 0.080759 | `distribution_scores.py` |
| 多水平反例 | 50% 原始覆盖 0.420(1年) vs 0.370(15年);TCN 过覆盖 0.930/0.922 | `training_summary.py` |
| 匹配 365−180 天记忆 NSE 效应 | 短记录 +0.0002 [-0.0022,0.0070];长记录 +0.0072 [0.0002,0.0125] | `training_summary.py` |

评分与点技巧方向比原始覆盖更可推广;不宣称普遍的原始覆盖梯度。方法身份重要:遗留产物标识 `umal`/`encqr` 分别指高斯位置尺度分位头与逐成员 split-CQR 平均,不是原引用的 UMAL/EnCQR 算法。

## 目录结构

```
src/                    库层(只被 import,不直接运行)
├─ evaluation/          评价协议:CQR 半径、区间评分、IQS、解析 CRPS、分组、空集投影
├─ models/              LSTM(LPU-Stream)、标准 LSTM、TCN
├─ heads/               稠密分布头(MDN、高斯、噪声)
├─ benchmark/           671 流域基准加载器
├─ corrections/         修正后稀缺实验共享加载器
├─ data/                CAMELS 读取、预处理、标准化
└─ losses/              pinball 损失、CQR 校准器

experiments/            运行层,每脚本一个职责
├─ prepare/             缓存构建/校验、数据链与检查点审计
├─ train_primary/       主 12 拟合、互斥子集、端点复跑
├─ train_benchmark/     671 流域八种构造
├─ bias_correction/     中位数偏差修正诊断
├─ train_sensitivity/   48 个端点拟合(队列驱动)
├─ evaluate/            覆盖、评分、方法比较、校准变体
├─ analyze/             汇总、配对效应、代表性、学习曲线
└─ figures/             论文全部图

run_pipeline.py         唯一总入口,按阶段顺序调度
download_camels.py      下载 CAMELS-US 数据档案
tests/                  协议单元测试(无需数据集)
configs/                数据配置
```

依赖单向:`run_pipeline.py → experiments/* → src/*`;脚本之间不互相 import,共享逻辑全部在 `src/`。

## 复现

```bash
pip install -r requirements.txt
python download_camels.py
python src/data/data_preprocessing.py
python run_pipeline.py --list      # 查看阶段
python run_pipeline.py --dry-run   # 只解析脚本路径,不运行
python run_pipeline.py             # 全量重跑(阶段 2/3/5 需要 CUDA GPU)
python -m pytest tests -q          # 协议测试
```

参考环境:Python 3.11、PyTorch 2.11.0+cu128、RTX 5060 Ti(8GB);见
`requirements-reference-cu128.txt`。数据、权重与逐样本预测不入库,各阶段在
`results/` 下重建产物;固定提交轮快照及其 SHA-256 清单见
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 引用与许可

见 [CITATION.cff](CITATION.cff) 与 [LICENSE](LICENSE)。早期发布系列有 Zenodo
概念 DOI [10.5281/zenodo.21256955](https://doi.org/10.5281/zenodo.21256955),
该概念记录不标识本轮修订。
