# Person A 数据交接

已完成 50 首 MAESTRO v3.0.0 钢琴独奏录音的预处理。

## 交给 Person B / C 的文件

- `data/problem2/processed/train.npy`：float32，形状 **(1129, 16, 12)**，来自官方 train 的 40 首录音。
- `data/problem2/processed/test.npy`：float32，形状 **(297, 16, 12)**，来自官方 test 的 10 首录音。
- `data/problem2/processed/manifest.json`：逐曲统计、参数、每个窗口的录音名/起止时间/原始 beat 索引，以及数组中的行索引。
- `results/problem2/data/real_chroma_heatmap.png`：3 个真实训练样本的热力图。
- `results/problem2/data/beat_tracking_spotcheck.png`：3 首训练录音的节拍与 onset strength 抽查。
- `results/problem2/data/song_summary.csv`：50 首录音的处理统计。
- `results/problem2/data/validation.json`：数值验证结果。

从项目根目录加载：

```python
import numpy as np
train = np.load('data/problem2/processed/train.npy')
test = np.load('data/problem2/processed/test.npy')
```

Person B 只用 train 训练；Person C 用 test 评价。每个样本展平为 192 维后可直接进入现有 MLP。
Notebook 的真实数据分支已改成加载这些数组，不再每次重新提取音频特征。
Person A 的数据准备阶段不包含模型训练。后续已完成一次正式训练与 A/B/C/D 评价
（截至 2026-10-11）：Notebook 保存了 `REAL_EXPERIMENT_OK`，权重位于
`models/problem2/flow.pt`，样本位于 `results/problem2/samples.npz`，
指标、配置和图表位于 `results/problem2/`。B/C 可先复核这些结果，再完成报告；
从头运行 Notebook 会重新训练并覆盖对应正式产物。

## 预处理规则

音频为完整录音的 22050 Hz 单声道 FLAC。用 librosa 默认 beat tracker 检测边界，
`chroma_stft(norm=None)` 提取 12 维 chroma，hop_length=512。只在相邻检测 beat 边界之间
平均帧级 chroma，开头/结尾没有完整 beat 边界的部分不进入样本。
音高类别顺序：C, C#, D, D#, E, F, F#, G, G#, A, A#, B。

低能量阈值为每首录音 `max(1e-5, peak_frame_RMS * 0.001)`，即相对峰值 RMS 的 -60 dB，
或绝对 RMS 下限。beat 内均方根能量低于该阈值，或 chroma 总能量接近零时，标记为无效。
有效 beat 按 12 维总和做 L1 归一化；无效 beat 保留其原始位置。

从首个 beat 区间开始切分不重叠 16-beat 窗口；任一 beat 无效就排除整个窗口，
不删除 beat 后重新拼接。训练集发现 12 个无效 beat，排除 5 个窗口；测试集没有触发此过滤。
训练/测试各丢弃不足 16 beats 的尾部合计 329 / 92 beats。
阈值预先固定，没有利用测试集调整。

## 数据划分与可复现性

读取 `sources.json` 的 MAESTRO 官方标签：40 train、10 test，不重新随机划分。
核验所有源 FLAC 的 SHA256，检查跨 split 的相同文件和完全相同作曲家/标题字符串。
作品标题可能存在别名，因此不能把 50 首录音称为 50 个独立作品；同作品不同演奏的归组依赖官方划分。
`manifest.json` 记录源清单和预处理代码哈希。音频、源码或规则改变后，重新运行：

```text
python scripts/prepare_person_a.py
python -m unittest discover -s tests -v
```

## 检查结果与限制

- 两个数组均为 float32，无 NaN/Inf、无负值。
- 逐拍总和误差最大约 2.98e-7（训练）、2.38e-7（测试），低于 1e-5 容差。
- 50 首都有有效窗口，零整首跳过，训练与测试文件没有交集。
- 每个输出窗口都核验不含无效 beat，保留连续的原始 beat 索引。
- 4 项回归测试通过，涵盖官方标签、跨 split 重复、音频改动、静音窗口不跨段拼接及模型基础通路。

节拍抽查选择训练集 beat 间隔变异系数最低、中位和最高的三首，
分别为 maestro_034、maestro_038、maestro_017；每首展示约 12 秒。
图中不少 beat 与音头峰值相近，但弱起、长音及停顿处存在不对齐；
尤其 038 的约 4–5 秒和 017 的部分弱音区域，需要保留谨慎解释。
这是有限的视觉抽查，没有人工或乐谱节拍真值，无法排除半速/倍速、rubato 和重音层级误差。
没有因为图形看起来不理想而事后筛除测试数据。

每首窗口数不均，窗口也并非独立作品。MVP 的默认训练/评价以窗口为单位，
因此较长或较快录音可能占更高权重。报告应限定为小规模 pilot study。

数据来源、许可及曲名详见 `DATASET.md`、`playlist.csv` 和 `sources.json`；
使用 MAESTRO 时保留 CC BY-NC-SA 4.0 署名与引用。Problem 1 未改动。
