"""Check a fresh checkout without downloading audio or training a model."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import VelocityNet
from src.constraints import project_simplex
from src.evaluation import feasibility
import torch

processed = ROOT / 'data/problem2/processed'
manifest = json.loads((processed / 'manifest.json').read_text())
assert manifest['version'] == 2 and 'official' in manifest['split_policy']
for name, key in [('src/data.py', 'preprocessing_sha256'),
                  ('data/problem2/sources.json', 'source_manifest_sha256')]:
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == manifest[key], f'Rebuild data: {name} changed'
for split in ['train', 'test']:
    x = np.load(processed / f'{split}.npy', allow_pickle=False)
    assert x.shape == (len(manifest['windows'][split]), 16, 12)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert (x >= 0).all() and np.allclose(x.sum(-1), 1, atol=1e-5)
    assert feasibility(x)['feasible_sequences'] == 1
    print(split, x.shape, x.dtype)
model = VelocityNet()
assert model(torch.zeros(2,192), torch.zeros(2,1)).shape == (2,192)
assert torch.allclose(project_simplex(torch.zeros(2,12)).sum(-1), torch.ones(2))
print('CHECKOUT_READY: data and model interfaces work without raw audio')
