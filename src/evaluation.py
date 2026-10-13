"""Feasibility and distribution metrics for raw chroma samples."""
import numpy as np


def _array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (16,12) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite (N,16,12)')
    return x


def feasibility(x, tol=1e-5):
    x = _array(x)
    negative = np.maximum(-x,0).sum(-1)
    error = np.abs(x.sum(-1)-1)
    valid = (x.min(-1) >= -tol) & (error <= tol)
    return dict(negative_mass_mean=float(negative.mean()), sum_error_mean=float(error.mean()),
                sum_error_max=float(error.max()), feasible_beats=float(valid.mean()),
                feasible_sequences=float(valid.all(-1).mean()))


def temporal_stats(x):
    x = _array(x)
    changes = np.linalg.norm(np.diff(x,axis=1),axis=-1)
    return dict(beat_l2_mean=float(changes.mean()), beat_l2_std=float(changes.std()),
                beat_l2_p95=float(np.quantile(changes,.95)))


def _dist(x,y):
    return np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0)


def reference_bandwidth(reference, max_samples=512, seed=42):
    """Median positive squared distance; fit once on held-out reference only."""
    x = _array(reference).reshape(len(reference),-1)
    x = x[np.random.default_rng(seed).permutation(len(x))[:max_samples]]
    d = _dist(x,x); values = d[np.triu_indices(len(x),1)]
    values = values[values > 1e-12]
    return float(np.median(values)) if len(values) else 1.0


def mmd_squared(x,y,bandwidth_squared):
    """Biased RBF MMD^2 including diagonals; capped by caller, O(N^2)."""
    if bandwidth_squared <= 0:
        raise ValueError('bandwidth_squared must be positive')
    x,y = _array(x),_array(y)
    x,y = x.reshape(len(x),-1),y.reshape(len(y),-1)
    kernel = lambda a,b: np.exp(-_dist(a,b)/(2*bandwidth_squared))
    return float(max(0, kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()))


def evaluate(x,reference,bandwidth_squared):
    stats, target = temporal_stats(x), temporal_stats(reference)
    return {**feasibility(x), 'mmd2': mmd_squared(x,reference,bandwidth_squared),
            **stats, 'beat_l2_mean_gap': abs(stats['beat_l2_mean']-target['beat_l2_mean'])}
