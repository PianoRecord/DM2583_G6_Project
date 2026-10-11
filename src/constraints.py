"""Member C owns projection; coordinate sampler changes with member B."""
import torch


def project_simplex(x):
    """Euclidean projection onto {x >= 0, sum(x) = 1} along last axis.

    Sorting/threshold algorithm; NOT clipping followed by normalization.
    """
    if not x.is_floating_point() or x.shape[-1] < 1 or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating-point tensor')
    u = x.sort(dim=-1, descending=True).values
    cssv = u.cumsum(-1) - 1
    ranks = torch.arange(1, x.shape[-1]+1, device=x.device, dtype=x.dtype)
    rho = (u - cssv / ranks > 0).sum(-1, keepdim=True).clamp_min(1)
    theta = cssv.gather(-1, rho-1) / rho.to(x.dtype)
    return (x-theta).clamp_min(0)


def correct_step(x, method, alpha=0.1):
    if method not in 'ABCD' or len(method) != 1 or not 0 <= alpha <= 1:
        raise ValueError('method must be A/B/C/D and alpha in [0,1]')
    if method == 'B':
        return x + alpha * (project_simplex(x)-x)
    if method == 'C':
        return project_simplex(x)
    return x
