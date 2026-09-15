import unittest
import tempfile
from pathlib import Path

import torch

from strong_rf.common import integrate, to_uint8, build_model, build_prior
from unified_transport.ot_pairing import pair_latents_by_class
from strong_rf.train import reverse_ce_loss, atomic_save


class LinearVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate=torch.nn.Parameter(torch.tensor(.2))
        self.seen_labels=[]
    def forward(self,t,x,y=None):
        self.seen_labels.append(y)
        return self.rate*x


class StrongRFTests(unittest.TestCase):
    def test_checkpoint_rotation_keeps_two_distinct_recovery_points(self):
        with tempfile.TemporaryDirectory() as folder:
            latest=Path(folder)/'training-latest.pt'
            for step in [25000,50000,75000]:
                atomic_save({'step':step},latest,keep_previous=True)
            self.assertEqual(torch.load(latest)['step'],75000)
            self.assertEqual(torch.load(Path(folder)/'training-previous.pt')['step'],50000)
            self.assertEqual(len(list(Path(folder).glob('*.pt'))),2)

    def test_center_ce_reverse_gradient_and_posterior_decision(self):
        prior = build_prior({'kind':'gaussian_ce','sigma':.5}, 'cpu')
        model = LinearVelocity()
        labels = torch.tensor([0,1])
        # Both examples initially favor the wrong fixed Gaussian center.
        x = prior.centers[torch.tensor([1,0])].clone()
        z, _ = integrate(model, x, None, 'euler', 4, reverse=True)
        self.assertTrue(torch.equal(prior.classify(z), prior.center_coordinates(z).argmax(1)))
        loss = reverse_ce_loss(prior, z, labels, 'center')
        loss.backward()
        self.assertTrue(torch.isfinite(model.rate.grad))
        self.assertGreater(float(model.rate.grad.abs()), 0.)
        self.assertTrue(all(y is None for y in model.seen_labels))

    def test_reverse_ode_and_parameter_gradient(self):
        model=LinearVelocity()
        x=torch.ones(2,3,2,2)
        z,nfe=integrate(model,x,None,'heun',100,reverse=True)
        self.assertEqual(nfe,200)
        self.assertTrue(torch.allclose(z,torch.exp(-model.rate)*x,atol=2e-6))
        z.mean().backward()
        self.assertAlmostEqual(float(model.rate.grad),-float(torch.exp(-model.rate)),places=5)
        self.assertTrue(all(y is None for y in model.seen_labels))
        with torch.no_grad():
            roundtrip,_=integrate(model,z,None,'heun',100)
        self.assertTrue(torch.allclose(roundtrip,x,atol=2e-6))

    def test_quantization_matches_authors(self):
        x=torch.tensor([-2.,-1.,0.,1.,2.])
        self.assertEqual(to_uint8(x).tolist(),[0,0,128,255,255])

    def test_ot_preserves_each_sample_and_class(self):
        labels=torch.tensor([0,0,1,1,1,2])
        prior=build_prior({'kind':'gaussian_ot','sigma':.5},'cpu')
        images=torch.randn(6,3,32,32)
        torch.manual_seed(42);original=prior.sample(labels)
        torch.manual_seed(42);paired,_=pair_latents_by_class(images,labels,prior,epsilon=.005,iterations=10)
        for c in labels.unique():
            a=original[labels==c].flatten(1);b=paired[labels==c].flatten(1)
            matches=(a[:,None,:]==b[None,:,:]).all(-1)
            self.assertTrue(bool((matches.sum(0)==1).all()))
            self.assertTrue(bool((matches.sum(1)==1).all()))

    def test_seed_matched_trunk(self):
        torch.manual_seed(3);a=build_model('gaussian')
        torch.manual_seed(3);b=build_model('label')
        for name,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        self.assertIsNone(a.num_classes)
        self.assertEqual(b.num_classes,10)


if __name__=='__main__':unittest.main()
