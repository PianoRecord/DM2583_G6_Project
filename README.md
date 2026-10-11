# DM2583 Group 6 — Problem 2 MVP

目标：2–3 天完成 MAESTRO v3.0.0 的 50 首钢琴独奏录音的 normalized beat-chroma 生成实验。每个样本
`(16,12)`，展平为 192 维。复用 Problem 1 的三层 128 单元 SiLU MLP 和线性
Flow Matching 训练方式，重新训练音乐模型（不加载 torus 权重）。

## 队友 clone 后的最短运行流程

推荐 Python 3.11。从项目根目录运行：

```text
python -m venv .venv
```

macOS/Linux 激活：`source .venv/bin/activate`；Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`。
然后：

```text
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m unittest discover -s tests -v
```

在 PyCharm/VS Code 中选择新建 `.venv` 作为 Notebook 内核，打开 `problem2.ipynb`，
保持 `SMOKE=False`，从头运行即可训练与评价。这里的环境依赖已在原开发机验证；
没有实际测试所有操作系统的新环境安装。

**B/C 无需下载音频：** `data/problem2/processed/train.npy` 和 `test.npy` 已被明确允许入库，
连同 manifest、src、Notebook、说明及图表一起提交后，足以继续训练/评价。
截至 2026-10-11，已完成一次真实数据正式训练与 A/B/C/D 评价；Notebook 保存了
`REAL_EXPERIMENT_OK`。正式权重为 `models/problem2/flow.pt`，生成样本、共享噪声和
参考数据保存在 `results/problem2/samples.npz`，指标、配置和图表位于同目录。
本次使用 1129 个训练窗口、297 个测试窗口，训练 3000 步，采样 100 步。
Notebook 从头运行会重新训练并覆盖对应正式产物，不会自动加载现有权重；可先查看已有结果。

**只有需要重做 Person A 时才下载音频：**

```text
python scripts/download_maestro_subset.py
python scripts/prepare_person_a.py
```

下载按官方 ZIP 范围请求取得约 1.63 GB 数据，转换后的音频约 189 MB。
它们不会进入普通 Git 提交。来源、许可及固定曲目清单均保留在 `data/problem2/`。
`SMOKE=True` 只用于合成数据排错，不能代替正式实验。

## 本次提交应包含什么

将 `.gitignore`、`.gitattributes`、README、requirements、`src/`、`scripts/`、`tests/`、
`problem2.ipynb`、`data/problem2/` 内的说明/清单/processed 数组，以及
`results/problem2/data/` 的数据检查结果一起纳入提交。
不要只提交已跟踪文件的改动，否则新模块和数据会遗漏。
原始音频、虚拟环境、缓存和 smoke 输出保持忽略。

注意：`.idea/` 已在旧提交中跟踪，`.gitignore` 不能自动将它从历史或跟踪列表移除。
本次没有调整 Git 暂存区；如要清理这些 IDE 配置，可另行操作，不影响 B/C 数据使用。

## Person A 数据交接

运行 `python scripts/prepare_person_a.py` 可重建真实数据与检查图，不训练模型。
已准备的数据在 `data/problem2/processed/`；详情见 `data/problem2/PERSON_A_HANDOFF.md`。
`results/problem2/data/` 保存热力图、节拍抽查图、逐曲统计和验证记录。

## 接口与实验约定

- A 负责 `src/data.py` 和数据来源记录：音频→beat-synchronous STFT chroma→每拍 L1 归一化。
  MAESTRO 子集读取 sources.json 的官方 40/10 划分，不重新随机分割，再构造不重叠 16 拍窗口，丢弃不足 16 拍的尾部。
  不使用全局标准化；没有测试集拟合的归一化参数。按每首录音的 RMS 阈值标记低能量 beat，丢弃包含它们的整个窗口；不跨静音拼接。整首静音或 beat 不足则记录跳过。
  输出 `train.npy/test.npy` 为 float32 `(N,16,12)`，`manifest.json` 记录歌曲、数量、跳过原因和配置。
  核验来源音频 SHA256 和跨 split 的重复文件/同名作品；作品标题别名由官方划分处理。
- B 负责 `src/model.py`、训练及最终 notebook 集成：CPU，默认 3000 steps、batch 256。
- C 负责 `src/constraints.py`、`src/evaluation.py`、图表与报告。
  B/C 先约定投影接口；避免同时改同一文件。Notebook 由 B 汇总，A/C 通过模块提供结果。

约束集合是每拍 12 维概率 simplex。A 无约束；B 每个 Euler 步后
`x ← x + 0.1*(projection(x)-x)`；C 每步完整 Euclidean simplex projection；
D 仅终点投影。B 的 alpha 固定为每步 0.1，不等同于 Problem 1 的连续强度参数。
固定 100 个 Euler 步，共用同一模型、同一初始 Gaussian noise；不投影初始噪声。

评价原始生成输出，不对 A/B 额外归一化：负值质量、和误差、可行比例；
flattened sequence 的 biased RBF MMD²（含对角线），最多 512 个样本，
同一 held-out reference 和固定 median-distance bandwidth；相邻 beat L2 均值/标准差/P95。
时间统计只在窗口内部计算，不跨窗口拼接。MMD 和时间统计衡量表示分布，不代表听感质量。

## 三人协作 / 2–3 天

- Day 1：A 准备 train/test 与来源清单；B 调通 192 维模型；C 验证 projection 和指标。
- Day 2：一次有效训练，生成 A/B/C/D；产出指标 CSV 和 3 张核心图；共同检查结论。
- Day 3（若有）：排错、报告、讨论，不再增加新功能。
- 每人使用独立分支；同步 main 后再开始；提交时只包含代码、说明和必要的小型结果。
  大型音频、权重、环境不入库。`.gitignore` 不会删除或停止跟踪已入库文件。
- 本次只改本地文件；提交、推送和仓库协作操作由组员自行完成。

## 文件与结果

`src/` 放可复用模块；`problem2.ipynb` 串联实验；`tests/` 放短 smoke tests。
真实运行输出到 `results/problem2/`，权重到 `models/problem2/`；
合成 smoke 结果单独放在 `results/problem2/smoke/`，不会覆盖正式权重。
现有空 `Problem2/` 不另建一套实现，统一采用根目录 notebook + src 布局。

**范围限制：** simplex 仅保证表示有效，不保证和声合理。数据量小、单次训练、
beat tracking 误差、歌曲窗口数量不均都会影响结论。更多数据集、多训练种子、
和声连续性约束和听感评价作为展望；不能预设 C 或 D 一定更好。

---

# DM2583 Project — Problem 1

已完成 `problem1.ipynb`：R=3、r=1 的 torus、Flow Matching 训练、四种 constraint handling、feasibility / fidelity 指标、可视化及小型敏感性实验。

## 运行

在 PyCharm 中打开桌面 `DM2583_Project` 项目，选择项目 `.venv/bin/python` 作为 notebook Python 环境，打开 `problem1.ipynb` 后从头运行全部单元。Notebook 默认重新训练，结果保留在单元输出中。

依赖：torch、numpy、pandas、matplotlib、ipykernel、nbformat、nbclient。已在项目 `.venv` 中补装 PyTorch。

## 实验约定

课程源文件：桌面 `DM2583_BigData/project.pdf`（Course Project 2026）。说明要求同一训练模型比较 A/B/C/D，规定 R=3、r=1，但没有明确表面采样密度。本实现采用**表面积均匀**分布，并解释其角度密度；不是两个角度都均匀的分布。

- 数据：20000 个训练样本，4000 个独立评估样本。
- 训练：6000 步，batch size 512，3 层 128 单元的 MLP，CPU。
- 采样：100 个 Euler 步；soft 修正强度 5；可行性距离阈值 0.01。
- Fidelity：128 方向 3D sliced Wasserstein-1，以及 12×12 联合角度分箱相对解析密度的 total variation。
- 三个额外采样种子只评估采样波动；没有重训多个模型。

## 生成文件

- `problem1.original.ipynb`：修改前的 PyCharm 示例备份。
- `data/torus_train.npy`、`data/torus_reference.npy`：训练与评估数据。
- `models/torus_flow_matching.pt`：模型参数。
- `results/problem1/`：指标 CSV、采样点、配置及 PNG 图形。

指标与结论以实际执行输出为准。不要预设某一种约束方式一定最佳。
