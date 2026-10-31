[English](README.md) | [简体中文](README.zh-CN.md)

# 监督训练记录长度对径流预报点预测技巧与区间可靠性的影响

本项目对应JOH大修稿。研究对象为**固定独立校准记录下的训练标签不足**，一年训练仍需额外校准观测，不代表只有一年全部径流观测的流域。

主实验为固定50流域、1/3/5/15年、每条件三个模型种子；其余四个互斥50流域子集只复现1/15年端点，合计250独立流域。记录长度也改变年份组成，固定epochs改变更新数，不能当作纯年数因果识别。

主样本NSE中位数一年0.511、十五年0.666，配对效应0.140（95%流域区间[0.104,0.173]），94%流域为正。原始90%池化覆盖0.776→0.821；等流域配对差4.6个百分点（[1.5,7.7]），仅72%流域为正。短记录区间更宽且评分更差。全局CQR使池化覆盖接近名义值，但不保证各组可靠。

校准预测阈值事前定义的高预测流量组覆盖0.832→0.882，流域、HUC2与同步水文年区间均支持正增益。实测高流量组为事后选样，即使条件分位数完全正确也不必达到90%，两类指标分别解释。

本轮新增48次拟合：18次替代年份、9次独立验证、6次多水平、6次TCN、9次匹配目标日期的180/365天记忆实验。反例保留：TCN两端均过覆盖，13分位50%水平覆盖方向反转；点技巧与评分方向更一致，不声称普遍原始覆盖梯度。

本轮修正：

- 原分布评分混用标准化观测与还原后预测，现全部统一log-flow。共同三点/稠密13点评分称有限网格IQS；高斯解析完整CRPS0.080759，13点IQS0.078314。
- 旧`umal`实际为高斯位置尺度分位头；旧`encqr`为静态逐成员split-CQR平均，不声称实现所引原算法，旧ID只追踪数组。
- 实测分组统一流域内0.33/0.67阈值、重复值归低组、严格高于上阈值；预测组统一流域内校准预测阈值和相对评分池。
- `src/evaluation_protocol.py`统一有符号评分、有限样本秩、空集投影和评分。区间以历史时期与固定权重为条件，与模型种子SD区分。

冻结投稿代码/紧凑结果包标识`JOH-revision-20261030`，清单在`revise/release/`。这不表示已经公开新版本。直接[GitHub仓库](https://github.com/AuroraNeko/data-scarcity-uncertainty-hydrodl)及旧[Zenodo concept DOI](https://doi.org/10.5281/zenodo.21256955)均明确给出，后者不能单独标识本轮文件。大预测与权重实际可用性单独说明。

复现顺序见[revise/REPRODUCE.md](revise/REPRODUCE.md)和[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。参考环境Python3.11、PyTorch2.11.0+cu128、RTX5060Ti8GB。数据、权重与逐样本预测不进普通Git；缺预测归档时仅凭紧凑表不能重算逐样本指标。旧`revision_work/`保留来源追踪，本轮权威评价入口为`revise/code/`。公开代码包不含作者稿和审稿通信。

```bash
pip install -r requirements.txt
python revise/code/prepare_training_cache.py
python revise/code/run_training_queue.py
python revise/code/analyze_revision_training.py --require-complete
python revise/code/publication_outputs.py --require-complete
python -m unittest discover -s tests -v
```
