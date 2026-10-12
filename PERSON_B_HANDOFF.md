# Task B notes

The model was trained on the arrays prepared in task A. The preprocessing code
and data files are unchanged.

## Files

Task B covers:

- `src/model.py`: the 192-dimensional velocity network, Flow Matching training,
  and Euler sampling.
- Training and A/B/C/D sampling in `problem2.ipynb`.
- The local model checkpoint and generated sample archive.

Inputs:

- `data/problem2/processed/train.npy`: `(1129, 16, 12)`, `float32`.
- `data/problem2/processed/test.npy`: `(297, 16, 12)`, `float32`.
- `data/problem2/processed/manifest.json` and the source records from task A.
- `src/constraints.py` and `src/evaluation.py`, which are owned by Person C.

`scripts/check_setup.py` checked the hashes in the manifest, the official split,
array shapes, finite values, non-negativity, and per-beat sums. Training uses
these cached arrays, so there is no need to download or process audio again.

## Training

Each `(16, 12)` sample is flattened to 192 values. The network takes those values
and time `t` as input. It has three hidden layers of 128 units with SiLU
activations and predicts a 192-dimensional velocity.

For every training step:

1. Draw `x0 ~ N(0, I)` and a real sample `x1`.
2. Draw `t ~ Uniform(0, 1)`.
3. Form the linear interpolation `xt = (1 - t) * x0 + t * x1`.
4. Train the network to predict the target velocity `x1 - x0` with mean squared
   error.

The run used seed 42, 3000 steps, batch size 256, Adam with learning rate
`1e-3`, and CPU execution. The first loss was 1.02014 and the final loss was
0.55413; the mean of the first 100 losses was 0.94815 and the mean of the last
100 was 0.57230. Repeating training with the same configuration reproduced the
saved checkpoint exactly.

## Sampling

All four methods use the same trained model, the same 297 initial Gaussian
noise tensors, and 100 Euler steps. The initial noise is not projected.

- A: Euler update only; no constraint correction.
- B: after every Euler update, apply
  `x <- x + 0.1 * (project_simplex(x) - x)`.
- C: after every Euler update, fully project each 12-dimensional beat onto the
  probability simplex.
- D: run the same unconstrained trajectory as A, then project only the final
  sample.

The saved D samples equal `project_simplex(A)` within
`1e-6`. C and D both have zero negative mass and 100% feasible beats and
sequences within the `1e-5` evaluation tolerance.

## Saved results

- `models/problem2/flow.pt`: model state and training/sampling configuration.
- `results/problem2/samples.npz`: reference data, reference indices, shared
  initial noise, and A/B/C/D arrays. Every generated array has shape
  `(297, 16, 12)` and contains only finite values.
- `results/problem2/training.png`: Flow Matching loss curve.
- `problem2.ipynb`: all cells ran with `SMOKE=False`; its final output is
  `REAL_EXPERIMENT_OK`.

Git ignores the checkpoint and `samples.npz`. Both files are saved locally.

## Checks (2026-10-12)

```text
python scripts/check_setup.py
    CHECKOUT_READY: data and model interfaces work without raw audio

python -m unittest discover -s tests -v
    Ran 4 tests - OK

jupyter execute --inplace --timeout=-1 problem2.ipynb
    completed successfully with REAL_EXPERIMENT_OK
```

The saved files were also checked:

- All `reference`, `noise`, and A/B/C/D arrays are finite.
- C and D satisfy the simplex constraint for every generated sequence.
- D is the final projection of A.
- A second deterministic training run produced exactly the saved weights.
- Task A's `src/data.py`, processed arrays, manifest, and source records were
  not modified.

## Running again

From the repository root, activate the project environment and run:

```text
python scripts/check_setup.py
python -m unittest discover -s tests -v
jupyter execute --inplace --timeout=-1 problem2.ipynb
```

Running all notebook cells retrains the model and overwrites the local
checkpoint, samples, metrics, and figures. Keep
`SMOKE=False` for the real experiment; `SMOKE=True` is only for synthetic code
checks.
