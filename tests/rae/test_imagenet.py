import copy
import itertools
import os
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image

from rae_imagenet.config import load_config, validate_config, training_signature
from rae_imagenet.data import ImageNetImages, StepBatchSampler
from rae_imagenet.flow import integrate, indexed_noise, learning_rate, fm_loss
from rae_imagenet.model import build_model
from rae_imagenet.state import EMA, atomic_save, rng_state, restore_rng
from rae_imagenet.train import update

RAE_REPO = os.environ.get('RAEV2_REPO', 'third_party/RAEv2')


class DataTests(unittest.TestCase):
    def test_imagefolder_counts_mapping_and_flip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ('train', 'val'):
                for name in ('n01440764', 'n01443537'):
                    folder = root / split / name; folder.mkdir(parents=True)
                    im = Image.new('RGB', (256,256), 'red')
                    im.paste('blue', (0,0,128,256)); im.save(folder/'one.png')
            with self.assertRaisesRegex(ValueError, 'full ImageNet'):
                ImageNetImages(root, 'train')
            ds = ImageNetImages(root, 'train', allow_subset=True)
            val = ImageNetImages(root, 'val', classes=ds.classes, allow_subset=True)
            a,y = ds[(0,False)]; b,y2 = ds[(0,True)]
            self.assertTrue(torch.equal(a.flip(-1),b)); self.assertEqual(y,y2)
            self.assertEqual(len(val),2)
            with self.assertRaisesRegex(ValueError, 'ordering'):
                ImageNetImages(root,'val',classes=list(reversed(ds.classes)),allow_subset=True)

    def test_resume_and_rank_shards_match_global_stream(self):
        def batches(start=0, rank=0, world=1):
            return list(StepBatchSampler(100,8,2,42,start,4,rank,world))
        full=batches()
        self.assertEqual(full[2*4:],batches(start=2))
        r0,r1=batches(rank=0,world=2),batches(rank=1,world=2)
        for step in range(4):
            local = r0[step*2:step*2+2]+r1[step*2:step*2+2]
            self.assertEqual(full[step*4:step*4+4],local)
        self.assertNotEqual(full[:4],full[4:8])


class FlowTests(unittest.TestCase):
    def test_cfg_endpoints_and_forward_euler_direction(self):
        class Oracle(torch.nn.Module):
            def forward(self,x,t,labels):return labels[:,None,None,None].expand_as(x).float()
        x=torch.zeros(2,2,4,4);labels=torch.tensor([1,2])
        for scale in [0,1,2]:
            actual=integrate(Oracle(),x,labels,num_classes=3,steps=4,scale=scale)
            expected=(3+scale*(labels-3)).float()[:,None,None,None].expand_as(x)
            self.assertTrue(torch.equal(actual,expected))

    def test_null_index_is_num_classes_and_loss_has_gradient(self):
        class Recorder(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
            def forward(self,x,t,labels):self.labels=labels;return x*self.weight
        m=Recorder();loss=fm_loss(m,torch.randn(2,2,4,4),torch.tensor([0,999]),1000,1.)
        self.assertEqual(m.labels.tolist(),[1000,1000]);loss.backward()
        self.assertTrue(torch.isfinite(m.weight.grad));self.assertGreater(abs(float(m.weight.grad)),0)

    def test_sampling_noise_does_not_depend_on_microbatch(self):
        full=indexed_noise(range(7),(2,4,4),42,'cpu')
        parts=torch.cat([indexed_noise(range(i,min(i+2,7)),(2,4,4),42,'cpu') for i in range(0,7,2)])
        self.assertTrue(torch.equal(full,parts))

    def test_schedule_and_operational_config_changes(self):
        c=load_config();t=c['train']
        self.assertAlmostEqual(learning_rate(t['warmup_steps'],t),t['lr'])
        for s in [t['decay_steps'],t['decay_steps']*2]:self.assertAlmostEqual(learning_rate(s,t),t['min_lr'])
        changed=copy.deepcopy(c);changed['train']['max_steps']=900000
        self.assertEqual(training_signature(c),training_signature(changed))
        changed['train']['lr']*=2
        self.assertNotEqual(training_signature(c),training_signature(changed))
        with self.assertRaises(ValueError):validate_config(c,world_size=3)

    def test_serialized_optimizer_ema_rng_resume_repeats_next_update(self):
        torch.manual_seed(1)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(2,2,1);self.embedding=torch.nn.Embedding(1001,2)
            def forward(self,x,t,labels):return self.conv(x)+self.embedding(labels)[:,:,None,None]+t[:,None,None,None]
        model=Toy();ema=EMA(model,.9);opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        c=load_config()['train'];c.update(global_batch=4,microbatch=2)
        batches=[(torch.randn(2,2,4,4),torch.tensor([0,999])) for _ in range(2)]
        def one():
            result=update(model,lambda x:x,iter(batches),opt,c,1000,torch.device('cpu'))
            ema.update(model);return result
        one()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'latest.pt'
            atomic_save(dict(model=model.state_dict(),ema=ema.state,optimizer=opt.state_dict(),rng=rng_state()),path)
            first=one();expected=copy.deepcopy(model.state_dict());expected_ema=copy.deepcopy(ema.state)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);ema.load(saved['ema']);restore_rng(saved['rng'])
            second=one()
            self.assertEqual(float(first[0]),float(second[0]))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)
            for k,v in ema.state.items():self.assertTrue(torch.equal(v,expected_ema[k]),k)


@unittest.skipUnless((Path(RAE_REPO)/'src/stage2/models/DDT.py').exists(),'Run setup_vendor or set RAEV2_REPO')
class ModelTests(unittest.TestCase):
    def test_original_size_parameter_count_without_gpu_allocation(self):
        with torch.device('meta'):
            m=build_model(load_config()['model'],RAE_REPO)
        self.assertEqual(sum(p.numel() for p in m.parameters()),869589396)
        self.assertEqual(m.ctx_embedder.embedding_table.num_embeddings,1001)
        self.assertEqual(m.num_cond_tokens,12)

    def test_checkpointed_network_matches_forward_and_all_gradients(self):
        c=load_config()['model'];c.update(hidden_size=[32,64],depth=[1,1],num_heads=[4,4])
        torch.manual_seed(42)
        a=build_model(c,RAE_REPO,checkpointing=False)
        # Activate modulation and output paths; default zero output would hide errors.
        with torch.no_grad():
            for n,p in a.named_parameters():
                if 'adaln_modulation' in n or 'final_layer.linear' in n:p.normal_(0,.02)
        b=build_model(c,RAE_REPO,checkpointing=True);b.load_state_dict(a.state_dict())
        x=torch.randn(1,1024,16,16);t=torch.tensor([.4]);y=torch.tensor([999])
        aa=a(x,t,y);bb=b(x,t,y)
        self.assertTrue(torch.allclose(aa,bb,atol=1e-6))
        aa.square().mean().backward();bb.square().mean().backward()
        for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.requires_grad:
                self.assertIsNotNone(p.grad,n);self.assertIsNotNone(q.grad,n)
                self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-4),n)
        a.eval()
        with torch.no_grad():
            self.assertGreater(float((a(x,t,torch.tensor([0]))-a(x,t,y)).abs().mean()),0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
