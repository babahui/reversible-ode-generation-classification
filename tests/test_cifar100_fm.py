import unittest

import torch

from cifar100_fm.cifar100_prior import Cifar100LowFrequencyPrior, frequency_modes
from cifar100_fm.common import build_model, integrate, to_uint8
from cifar100_fm.run_study import milestones


class TestCifar100FM(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_codebook(self):
        prior = Cifar100LowFrequencyPrior()
        centers = prior.centers.flatten(1)
        self.assertEqual(len(set(frequency_modes())),100)
        self.assertTrue(torch.allclose(centers@centers.T,144*torch.eye(100),atol=.001))
        self.assertTrue(torch.equal(prior.classify(prior.centers),torch.arange(100)))

    def test_gaussian_and_posterior(self):
        prior = Cifar100LowFrequencyPrior()
        labels = torch.arange(100).repeat(2)
        z = prior.sample(labels,torch.Generator().manual_seed(123))
        residual = z-prior.centers[labels]
        self.assertLess(abs(float(residual.mean())),.003)
        self.assertLess(abs(float(residual.var())-.25),.003)
        distances = -torch.cdist(z[:5].flatten(1),prior.centers.flatten(1)).square()/.5
        self.assertTrue(torch.allclose(prior.log_posterior(z[:5]).softmax(1),
                                       distances.softmax(1),atol=1e-5))

    def test_solver_no_labels_roundtrip(self):
        class Constant(torch.nn.Module):
            def forward(self,t,x,y=None):
                if y is not None:
                    raise AssertionError('label leakage')
                return torch.ones_like(x)*2
        initial = torch.zeros(2,3,4,4)
        for method,multiplier in [('euler',1),('heun',2)]:
            endpoint,nfe = integrate(Constant(),initial,method=method,steps=8)
            reverse,_ = integrate(Constant(),endpoint,method=method,steps=8,reverse=True)
            self.assertEqual(nfe,8*multiplier)
            self.assertTrue(torch.equal(endpoint,initial+2))
            self.assertTrue(torch.equal(reverse,initial))

    def test_architecture(self):
        model = build_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()),35746307)
        self.assertIsNone(model.num_classes)

    def test_protocol_helpers(self):
        self.assertEqual(to_uint8(torch.tensor([-2.,-1.,0.,1.,2.])).tolist(),[0,0,128,255,255])
        self.assertEqual(milestones(125000,50000),[50000,100000,125000])


if __name__ == '__main__':
    unittest.main()
