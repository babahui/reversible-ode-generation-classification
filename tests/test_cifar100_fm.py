import unittest
from types import SimpleNamespace

import torch

from cifar100_fm.cifar100_prior import Cifar100LowFrequencyPrior, frequency_modes
from cifar100_fm.common import build_model, build_prior, integrate, to_uint8
from cifar100_fm.run_study import milestones, study_targets
from cifar100_fm.dual_view import paired_js_loss, paired_ce_images, auxiliary_loss, ramp_weight


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
        self.assertEqual(study_targets(1300000,100000,'gaussian_dual_pad4_js',1000000,
            [50000,100000]),[1050000,1100000,1200000,1300000])
        self.assertEqual(study_targets(5000000,100000,'gaussian',0,[])[-1],5000000)

    def test_paired_crop_rng_and_original_unchanged(self):
        images=torch.arange(4*3*32*32).reshape(4,3,32,32).float()
        original=images.clone()
        generator=torch.Generator().manual_seed(20260918)
        state=generator.get_state()
        global_state=torch.get_rng_state()
        views=paired_ce_images(images,4,generator)
        self.assertTrue(torch.equal(images,original))
        self.assertTrue(torch.equal(torch.get_rng_state(),global_state))
        self.assertEqual(tuple(views.shape),(8,3,32,32))
        self.assertFalse(torch.equal(*views.chunk(2)))
        generator.set_state(state)
        self.assertTrue(torch.equal(views,paired_ce_images(images,4,generator)))

    def test_js_and_temperature(self):
        scores=torch.randn(8,100,requires_grad=True)
        loss=paired_js_loss(scores,2.)
        self.assertGreaterEqual(float(loss),0.)
        self.assertLessEqual(float(loss),.693148)
        swapped=torch.cat(scores.chunk(2)[::-1])
        self.assertTrue(torch.allclose(loss,paired_js_loss(swapped,2.)))
        identical=torch.cat([scores[:4],scores[:4]])
        self.assertLess(abs(float(paired_js_loss(identical,2.))),1e-6)
        loss.backward()
        self.assertTrue(bool((scores.grad.abs().sum(1)>0).all()))

    def test_reverse_ce_js_has_no_label_input_and_gradients(self):
        class Field(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(.1))
            def forward(self,t,x,y=None):
                if y is not None or self.training:
                    raise AssertionError('CE reverse must disable dropout and labels')
                return self.weight*x
        args=SimpleNamespace(ce_batch=4,ce_crop_pad=4,ce_solver='euler',ce_steps=4,
            ce_js_temperature=2.,finetune_start=1000000,finetune_from='full.pt',
            ce_weight=.001,ce_warmup=2000,ce_js_weight=.001,ce_js_warmup=5000)
        model=Field().train()
        a,b,metrics=auxiliary_loss(model,Cifar100LowFrequencyPrior(),torch.randn(4,3,32,32),
            torch.tensor([0,99,20,30]),torch.Generator().manual_seed(5),args,1005000)
        self.assertTrue(model.training)
        (a+b).backward()
        self.assertTrue(bool(model.weight.grad.isfinite()))
        self.assertGreater(float(model.weight.grad.abs()),0.)
        self.assertAlmostEqual(metrics['weighted_js_loss'],.001*4*metrics['js_loss'],places=7)
        self.assertEqual(ramp_weight(.001,1000004,1000000,2000),.000002)

    def test_dual_model_uses_the_same_unconditional_backbone_and_prior(self):
        model=build_model('gaussian_dual_pad4_js')
        self.assertEqual(sum(p.numel() for p in model.parameters()),35746307)
        self.assertIsNone(model.num_classes)
        self.assertFalse(hasattr(model,'label_emb'))
        prior=build_prior(dict(kind='gaussian_dual_pad4_js',dataset='cifar100',num_classes=100,
            prior_version='rgb_dct6_energy_sorted_100_v1',center_scale=12.,sigma=.5),'cpu')
        self.assertEqual(prior.centers.shape,(100,3,32,32))


if __name__ == '__main__':
    unittest.main()
