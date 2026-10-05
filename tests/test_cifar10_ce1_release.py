"""Behavioral checks for Euler1 and checkpoint relocation/resume."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cifar10_ce1 import train
from cifar10_ce1.common import integrate


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,3,1)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self,t,x,y=None):
        return self.conv(self.dropout(x)) + t.reshape(-1,1,1,1)*.01


class CE1ReleaseTest(unittest.TestCase):
    def test_euler1_reverse_and_gradient(self):
        model = TinyVelocity().eval()
        images = torch.randn(2,3,32,32)
        encoded,nfe = integrate(model,images,method='euler',steps=1,reverse=True)
        torch.testing.assert_close(encoded,images-model(torch.tensor(1.),images))
        self.assertEqual(nfe,1)
        encoded.square().mean().backward()
        self.assertTrue(torch.isfinite(model.conv.weight.grad).all())
        self.assertGreater(model.conv.weight.grad.abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'Training kernel requires CUDA')
    def test_full_state_resume_after_path_relocation(self):
        dataset = types.SimpleNamespace(
            data=np.random.RandomState(4).randint(0,256,(40,32,32,3),dtype=np.uint8),
            targets=list(range(10))*4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(flags):
                with patch.object(train,'build_model',side_effect=lambda *a,**k:TinyVelocity()), \
                     patch.object(train,'CIFAR10',return_value=dataset), \
                     patch.object(sys,'argv',['train']+list(map(str,flags))), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()

            shared = ['--batch-size',2,'--save-every',2,'--snapshot-every',2,
                      '--preview-every',0,'--data-root',root/'data-a','--resume']
            run(shared+['--kind','gaussian','--output',root/'fm','--steps',4])
            source = root/'fm/training-latest.pt'
            ce = shared+['--kind','gaussian_ce','--ce-steps',1,'--ce-batch',4,
                         '--ce-logits','center','--ce-crop-pad',4,'--ce-crop-fraction',1,
                         '--ce-paired-views','--ce-paired-crop-both','--ce-js-weight',.001,
                         '--ce-warmup',2,'--ce-js-warmup',2,'--min-lr',.0002,
                         '--finetune-from',source,'--finetune-start',4,'--finetune-steps',8]
            run(ce+['--output',root/'continuous','--steps',12])
            run(ce+['--output',root/'split','--steps',8])
            moved = root/'relocated'
            shutil.copytree(root/'split',moved)
            moved_source = root/'relocated-source.pt'
            shutil.copy2(source,moved_source)
            run(ce+['--output',moved,'--steps',12,'--data-root',root/'data-b',
                    '--finetune-from',moved_source])
            a = torch.load(root/'continuous/training-latest.pt',map_location='cpu')
            b = torch.load(moved/'training-latest.pt',map_location='cpu')
            self.assertEqual(a['step'],b['step'])
            for field in ['model','ema']:
                for key,value in a[field].items():
                    torch.testing.assert_close(value,b[field][key],rtol=0,atol=0)
            for field in ['rng_cpu','rng_cuda','ce_crop_rng_state']:
                self.assertTrue(torch.equal(a[field],b[field]))
            for key,state in a['optimizer']['state'].items():
                for item,value in state.items():
                    torch.testing.assert_close(value,b['optimizer']['state'][key][item],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
