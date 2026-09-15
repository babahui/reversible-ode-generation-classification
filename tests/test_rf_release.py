import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from strong_rf import sample
from strong_rf.common import build_prior


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_wrapper_sets_center_ce_explicitly(self):
        env = dict(os.environ, KIND='gaussian_ot_ce', PYTHON='echo', RESUME='0')
        result = subprocess.run(['bash', 'scripts/train_strong_rf.sh'], cwd=ROOT,
                                env=env, check=True, capture_output=True, text=True)
        self.assertIn('--ce-logits center', result.stdout)
        self.assertIn('--ce-weight 0.001', result.stdout)
        self.assertIn('--sigma 0.5', result.stdout)

    def test_wrapper_refuses_missing_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, KIND='gaussian', PYTHON='echo', RESUME='1', OUTPUT=directory)
            result = subprocess.run(['bash', 'scripts/train_strong_rf.sh'], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing full resume checkpoint', result.stderr)

    def test_reverse_cli_passes_no_labels(self):
        prior = build_prior({'kind': 'gaussian'}, 'cpu')
        data = type('Dataset', (), {'data': np.zeros((2, 32, 32, 3), dtype=np.uint8),
                                    'targets': [0, 1], '__len__': lambda self: 2})()
        with tempfile.TemporaryDirectory() as directory:
            args = ['sample', '--checkpoint', 'dummy.pt', '--output', directory,
                    '--device', 'cpu', '--reverse-data-root', 'data', '--limit', '2']
            with patch('sys.argv', args), patch.object(sample, 'load_model', return_value=(
                object(), {'kind': 'gaussian'}, prior, 10)), patch.object(sample, 'sha256', return_value='test'), \
                 patch.object(sample, 'CIFAR10', return_value=data), patch.object(
                     sample, 'integrate', return_value=(prior.centers[:2], 120)) as solve:
                sample.main()
            self.assertIsNone(solve.call_args.args[2])
            self.assertTrue(solve.call_args.kwargs['reverse'])
            result = json.loads((Path(directory) / 'result.json').read_text())
            self.assertEqual(result['reverse_accuracy'], 1.)
            self.assertEqual(result['predictions'], [0, 1])

    def test_report_metrics_match_raw_protocol(self):
        rows = json.loads((ROOT / 'results/strong-rf/metrics.json').read_text())
        self.assertGreater(len(rows), 5)
        for row in rows:
            self.assertEqual(row['generated_images'], 50000)
            self.assertEqual(row['protocol']['steps'], 60)
            self.assertEqual(row['protocol']['method'], 'heun')
            self.assertEqual(len(row['protocol']['checkpoint_sha256']), 64)
            if row['kind'] != 'label':
                self.assertEqual(row['reverse_images'], 10000)
            else:
                self.assertNotIn('reverse_accuracy', row)
            raw = json.loads((ROOT / 'results/strong-rf' / row['evaluation'] / 'metrics.json').read_text())
            self.assertEqual(raw['fid_train_50k_reference'], row['fid_train_50k_reference'])


if __name__ == '__main__':
    unittest.main()
