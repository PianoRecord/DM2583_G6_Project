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
