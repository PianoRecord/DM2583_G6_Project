# DM2583 Group 6: constrained generation with Flow Matching

This project compares four ways of applying constraints to samples from a single
Flow Matching model. Problem 1 generates points on a torus. Problem 2 generates
16-beat normalized chroma sequences from MAESTRO piano recordings. Both experiments
measure constraint satisfaction (feasibility) and similarity to the target
distribution (fidelity).

## Installation

Python 3.11 is recommended. The Problem 2 experiment was also verified locally
with Python 3.12.14. Dependencies are listed in [requirements.txt](requirements.txt).
Run all commands from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install dependencies and
check the supplied data and interfaces:

```sh
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m unittest discover -s tests -v
```

## Running the experiments

Open [problem1.ipynb](problem1.ipynb) or [problem2.ipynb](problem2.ipynb) in a
Notebook editor and select `.venv` as the Python environment. Each notebook contains
the method, executed results, and discussion. To reproduce a full run:

```sh
jupyter execute --inplace --timeout=-1 problem1.ipynb
jupyter execute --inplace --timeout=-1 problem2.ipynb
```

Running all cells retrains the model and overwrites that experiment's checkpoint,
samples, metrics, and figures. The notebooks do not automatically load existing
weights. Problem 2 uses the prepared arrays in
[data/problem2/processed](data/problem2/processed), so it does not require raw audio.
Keep `SMOKE=False` for the real experiment. `SMOKE=True` runs a short synthetic-data
check and saves its outputs separately in `results/problem2/smoke/`.

## Methods and configuration

Training interpolates between independent Gaussian noise and real data:
`xt = (1 - t) * x0 + t * x1`. The MLP predicts `x1 - x0` using mean squared error.
Each experiment trains one model, then uses the same checkpoint, initial noise,
and Euler step count for all four sampling methods:

- A: no constraint correction.
- B: move partway towards the projection after each Euler step.
- C: project after every Euler step, starting from unprojected Gaussian noise.
- D: project only the final unconstrained sample; D equals the final projection of A.

The MLP has three hidden layers of 128 units with SiLU activations. Training runs
on CPU with Adam, learning rate `1e-3`, and seed 42.

| Setting | Problem 1 | Problem 2 |
| --- | --- | --- |
| Representation | 3D torus, R=3 and r=1 | 16 beats x 12 pitch classes, flattened to 192 values |
| Data | 20,000 training points; 4,000 evaluation points | 1,129 training windows; 297 test windows |
| Target / constraint | Uniform surface area / torus surface | Normalized chroma / per-beat probability simplex |
| Training | 6,000 steps, batch 512 | 3,000 steps, batch 256 |
| Euler sampling | 100 steps | 100 steps |
| Soft correction | `alpha = 1 - exp(-5 / steps)` | Fixed `alpha = 0.1` per step |
| Feasibility tolerance | Surface distance <= 0.01 | Negative entries and row-sum error within `1e-5` |

Problem 1 evaluates surface-distance statistics, 3D sliced Wasserstein-1 over 128
directions, and total variation in a 12 x 12 joint-angle histogram against the
analytical target density. Independent reference samples give a finite-sample
baseline. Three additional sampling seeds and a small step-count/soft-strength
sweep assess sampling variation; they do not retrain the model.

Problem 2 evaluates raw outputs without extra normalization: negative mass,
row-sum errors, and feasible beat/sequence fractions. Fidelity uses biased RBF
MMD² (including diagonals) on flattened sequences, with one shared held-out
reference and its median positive squared-distance bandwidth. Each group contains
297 samples, capped at 512. Adjacent-beat L2 distances provide a temporal statistic
(mean, standard deviation, P95, and mean gap from real data), calculated within
each window.

## Data and saved results

Problem 1 samples the torus uniformly by surface area using rejection sampling.
Problem 2 uses 50 full MAESTRO v3.0.0 recordings, with the official 40/10 train/test
split. See [DATASET.md](data/problem2/DATASET.md) for selection, preprocessing,
provenance, diagnostics, licensing, and instructions for rebuilding the audio data.

| Files | Contents |
| --- | --- |
| [data/torus_train.npy](data/torus_train.npy), [data/torus_reference.npy](data/torus_reference.npy) | Problem 1 training and reference points |
| [models/torus_flow_matching.pt](models/torus_flow_matching.pt) | Problem 1 checkpoint |
| [results/problem1](results/problem1) | Metrics, samples, configuration, figures, sampling replicates, and sensitivity results |
| [data/problem2/processed](data/problem2/processed) | Training/test arrays and the preprocessing manifest |
| [results/problem2/data](results/problem2/data) | Data validation, song statistics, chroma examples, and beat-tracking spot checks |
| [results/problem2](results/problem2) | Metrics, configuration, training curve, chroma comparison, and metric comparison |

Problem 2 also writes `models/problem2/flow.pt` and `results/problem2/samples.npz`
locally. Git ignores these two files, raw audio, virtual environments, and smoke
outputs. A fresh clone includes the prepared arrays, saved Notebook output,
metrics, and figures; running Problem 2 recreates its checkpoint and sample archive.
The archive contains A/B/C/D arrays, the shared initial noise, reference samples,
and reference indices. Each generated array has shape `(297, 16, 12)`.

The completed Problem 2 run passed data checks and all four tests. C and D have
100% feasible sequences within `1e-5`, and D matches the projection of A within
`1e-6`. Repeating training with the same configuration reproduced the saved weights.
Detailed numerical results remain in the metric files and notebooks.

## Limitations and references

Each experiment uses one trained model. Sampling replicates in Problem 1 do not
measure variation across training seeds. Model fitting, Euler discretization,
finite samples, and evaluation settings affect the results. The MAESTRO subset
covers solo piano only, windows are not independent musical works, and recordings
contribute unequal numbers of windows. Beat detection can be affected by rubato
or half/double-tempo errors. A simplex constraint guarantees a valid chroma
representation, not harmonic structure or listening quality. The results are a
small pilot study rather than a general ranking of constraint methods.

- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Hawthorne et al., *Enabling Factorized Piano Music Modeling and Generation with
  the MAESTRO Dataset*, ICLR 2019. MAESTRO attribution and CC BY-NC-SA 4.0 terms
  are recorded in [DATASET.md](data/problem2/DATASET.md).
