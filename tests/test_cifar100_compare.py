"""Synthetic fixtures test report behavior, not model quality."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from cifar100_fm.compare import collect, main


class CompareTests(unittest.TestCase):
    def fixture(self,root,kind,method='heun',count=10000):
        stage=root/'evaluations'/'fixture'
        stage.mkdir(parents=True)
        metric=dict(dataset='cifar100',kind=kind,checkpoint_step=1100000,
                    generated_images=50000,reverse_images=count,reverse_accuracy=.1,
                    fid_train_50k_reference=20.,inception_score_mean=3.)
        protocol=dict(method=method,steps=60,seed=20260912,batch_size=64,
                      samples=50000,reverse_images=10000,dataset='cifar100',num_classes=100)
        (stage/'metrics.json').write_text(json.dumps(metric))
        (stage/'protocol.json').write_text(json.dumps(protocol))
        Image.new('RGB',(340,340),'black').save(stage/'classes-00-09.png')

    def test_report_and_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root/'a','gaussian')
            self.fixture(root/'b','gaussian_dual_pad4_js')
            argv=['compare','--baseline',str(root/'a'),'--dual',str(root/'b'),
                  '--output',str(root/'out')]
            with patch.object(sys,'argv',argv):main()
            self.assertTrue((root/'out/comparison.csv').exists())
            self.assertIn('10.00%',(root/'out/README.md').read_text())
            with Image.open(root/'out/1100000-classes-00-09.png') as image:
                self.assertEqual(image.size,(692,364))

    def test_solver_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root/'a','gaussian')
            self.fixture(root/'b','gaussian_dual_pad4_js',method='euler')
            argv=['compare','--baseline',str(root/'a'),'--dual',str(root/'b'),
                  '--output',str(root/'out')]
            with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'protocol mismatch'):
                main()

    def test_partial_evaluation_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root,'gaussian',count=8)
            self.assertEqual(collect(root),{})


if __name__=='__main__':
    unittest.main()
