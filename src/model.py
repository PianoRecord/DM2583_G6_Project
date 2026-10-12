"""Flow matching for 16-beat chroma sequences."""
import torch
from torch import nn
from .constraints import correct_step, project_simplex


class VelocityNet(nn.Module):
    def __init__(self, dim=192, hidden=128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim+1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=1))


def train_flow(data, steps=3000, batch_size=256, lr=1e-3, seed=42, hidden=128):
    x = torch.as_tensor(data, dtype=torch.float32)
    if x.ndim != 3 or tuple(x.shape[1:]) != (16,12) or len(x)==0 or not torch.isfinite(x).all():
        raise ValueError('Expected nonempty finite (N,16,12) data')
    if steps < 1 or batch_size < 1:
        raise ValueError('steps and batch_size must be positive')
    torch.manual_seed(seed)
    x = x.reshape(len(x), -1)
    model = VelocityNet(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x1 = x[torch.randint(len(x), (batch_size,))]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1)
        loss = (model((1-t)*x0+t*x1, t)-(x1-x0)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
    return model.eval(), losses


@torch.no_grad()
def sample(model, noise, method='A', steps=100, alpha=0.1):
    if steps < 1 or method not in ('A','B','C','D') or not 0 <= alpha <= 1:
        raise ValueError('Invalid sampler settings')
    if noise.ndim != 3 or tuple(noise.shape[1:]) != (16,12):
        raise ValueError('Expected noise of shape (N,16,12)')
    model.eval()
    x = noise.detach().clone().to(next(model.parameters()))
    for k in range(steps):
        flat = x.reshape(len(x), -1)
        t = torch.full((len(x),1), k/steps, device=x.device, dtype=x.dtype)
        x = x + model(flat,t).reshape_as(x)/steps
        x = correct_step(x, method, alpha)
    return project_simplex(x) if method == 'D' else x
