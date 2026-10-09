"""Exercise metric persistence and interrupted CFG evaluation without GPU weights."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from rae_imagenet.config import load_config
from rae_imagenet.evaluate import evaluate, EvaluationInterrupted
from rae_imagenet.state import read_json


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()))
    def forward(self,x,t,labels):return x*self.weight


class Features:
    def __init__(self,*args):pass
    def __call__(self,images):
        return torch.stack([images.mean((1,2,3)),images.std((1,2,3))],dim=1).cpu(),torch.zeros(len(images),dtype=torch.long)


class Decoder:
    def __init__(self,*args):pass
    def __call__(self,z):return z.sigmoid()


def tiny_noise(indices,shape,seed,device):
    return torch.stack([torch.randn(3,4,4,generator=torch.Generator().manual_seed(seed+i)) for i in indices]).to(device)


class EvaluationTests(unittest.TestCase):
    def test_resume_partial_cfg_evaluation_and_reject_protocol_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);classes=['n01440764','n01443537']
            for c,name in enumerate(classes):
                folder=root/'data/val'/name;folder.mkdir(parents=True)
                for i in range(4):Image.new('RGB',(256,256),(30+c*50+i*10,100+i*15,120)).save(folder/f'{i}.png')
            args=SimpleNamespace(output=str(root/'out'),data_root=str(root/'data'),allow_subset=True,
                                 inception_weights=None,classifier_weights=None,rae_repo='unused',weights='unused')
            config=load_config();config['train']['num_workers']=0
            config['evaluation'].update(samples=1000,batch_size=100,decode_batch=100,steps=1,kid_subsets=2)
            calls=0
            def stop():
                nonlocal calls
                calls+=1
                return calls==12  # one reference batch + ten CFG1 batches; stop at CFG2
            target=root/'out/evaluations/step-000010000'
            with patch('rae_imagenet.evaluate.Features',Features),patch('rae_imagenet.evaluate.Decoder',Decoder),patch('rae_imagenet.evaluate.indexed_noise',tiny_noise):
                with self.assertRaises(EvaluationInterrupted):evaluate(Model(),config,args,10000,classes,stop=stop)
                self.assertTrue((target/'cfg-1/metrics.json').exists())
                self.assertFalse((target/'complete.json').exists())
                first_mtime=(target/'cfg-1/metrics.json').stat().st_mtime_ns
                evaluate(Model(),config,args,10000,classes)
                self.assertEqual(first_mtime,(target/'cfg-1/metrics.json').stat().st_mtime_ns)
                self.assertTrue((target/'complete.json').exists())
                score=read_json(target/'cfg-2/metrics.json')
                self.assertEqual(score['count'],1000);self.assertEqual(score['reference_count'],8)
                self.assertEqual(score['network_evaluations_per_image'],2)
                changed=copy.deepcopy(config);changed['evaluation']['seed']+=1
                with self.assertRaisesRegex(ValueError,'protocol changed'):evaluate(Model(),changed,args,10000,classes)


if __name__=='__main__':unittest.main()
