import unittest
import numpy as np
import torch
from src.data import normalize_chroma, make_windows, split_songs
from src.constraints import project_simplex
from src.model import train_flow, sample
from src.evaluation import feasibility, mmd_squared, reference_bandwidth, temporal_stats


class SmokeTests(unittest.TestCase):
    def test_data(self):
        x = normalize_chroma(np.zeros((33,12)))
        self.assertTrue(np.all(x == 0))
        self.assertEqual(make_windows(x).shape,(0,16,12))
        x = np.ones((49,12), dtype=np.float32)
        x[20] = 0
        windows, starts = make_windows(x, return_starts=True)
        self.assertEqual(starts.tolist(), [0,32])
        self.assertEqual(windows.shape, (2,16,12))
        self.assertTrue(np.allclose(windows.sum(-1),1))
        a,b = split_songs(['a','b','c','d','e'])
        self.assertFalse(set(a)&set(b))
        self.assertEqual((a,b),split_songs(['a','b','c','d','e']))

    def test_projection(self):
        x = torch.tensor([[.8,.4,-.2],[-2.,-2.,-2.]])
        expected = torch.tensor([[.7,.3,0.],[1/3,1/3,1/3]])
        self.assertTrue(torch.allclose(project_simplex(x),expected,atol=1e-6))
        z = torch.randn(20,16,12)*3
        p = project_simplex(z)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(20,16),atol=1e-5))
        self.assertTrue(torch.allclose(project_simplex(p),p,atol=1e-6))
        # Projection variational inequality against arbitrary feasible points.
        q = torch.softmax(torch.randn_like(z),-1)
        self.assertTrue((((z-p)*(q-p)).sum(-1)<=1e-4).all())

    def test_train_sample_metrics(self):
        torch.set_num_threads(2)
        data = np.random.default_rng(4).dirichlet(np.ones(12),size=(24,16)).astype('float32')
        model,loss = train_flow(data,steps=3,batch_size=8)
        noise = torch.randn(12,16,12); original=noise.clone()
        out = {m:sample(model,noise,m,steps=4) for m in 'ABCD'}
        self.assertTrue(torch.equal(noise,original))
        self.assertTrue(np.isfinite(loss).all())
        self.assertTrue(torch.allclose(out['D'],project_simplex(out['A']),atol=1e-6))
        self.assertTrue(torch.allclose(sample(model,noise,'B',4,0),out['A']))
        self.assertTrue(torch.allclose(sample(model,noise,'B',4,1),out['C'],atol=1e-6))
        for m in 'CD': self.assertEqual(feasibility(out[m])['feasible_sequences'],1)
        bw=reference_bandwidth(data)
        self.assertLess(mmd_squared(data,data,bw),1e-10)
        self.assertGreater(mmd_squared(data,data+1,bw),0)
        self.assertEqual(temporal_stats(np.ones((2,16,12))/12)['beat_l2_mean'],0)

if __name__ == '__main__': unittest.main()
