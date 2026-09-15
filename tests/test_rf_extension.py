import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from strong_rf import run_five_gpu as runner


def full_checkpoint(step=500000, kind='label'):
    return dict(step=step, config={'kind': kind}, model={'w': torch.ones(1)},
                ema={'w': torch.ones(1)}, optimizer={}, rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.get_rng_state(), rng_numpy=None)


class RFExtensionTests(unittest.TestCase):
    def test_requires_correct_full_continuation_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training-latest.pt'
            torch.save(full_checkpoint(), path)
            self.assertEqual(runner.validate_resume(path, 'label', 500000, 1000000), 500000)
            with self.assertRaises(ValueError):
                runner.validate_resume(path, 'gaussian', 500000, 1000000)
            with self.assertRaises(ValueError):
                runner.validate_resume(path, 'label', 550000, 1000000)
            torch.save({'ema': {}}, path)
            with self.assertRaises(ValueError):
                runner.validate_resume(path, 'label', 500000, 1000000)

    def test_selects_only_gpus_zero_and_one(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = Path(directory) / 'classifier.pt'
            classifier.touch()
            args = ['run', '--root', directory, '--cache', directory, '--target', '1000000',
                    '--classifier', str(classifier), '--start-step', '500000',
                    '--evaluation-samples', '50000', '--kinds', 'label', 'gaussian']
            with patch('sys.argv', args), patch.object(runner.subprocess, 'Popen') as launch, patch.object(runner, 'report'):
                launch.return_value.pid = 123
                launch.return_value.returncode = 0
                launch.return_value.poll.return_value = 0
                runner.main()
            self.assertEqual(launch.call_count, 2)
            self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in launch.call_args_list], ['0', '1'])
            for call in launch.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index('--start-step')+1], '500000')
                self.assertEqual(command[command.index('--target')+1], '1000000')

    def test_worker_continues_without_historical_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'label').mkdir()
            torch.save(full_checkpoint(), root / 'label/training-latest.pt')
            baseline = root / 'evaluations/label-500000-50k'
            baseline.mkdir(parents=True)
            (baseline / 'metrics.json').write_text(json.dumps({'generated_images': 50000, 'checkpoint_step': 500000}))
            commands = []

            class Child:
                pid = 123
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    if 'strong_rf.train' in command:
                        self_step = int(command[command.index('--steps')+1])
                        torch.save(full_checkpoint(self_step), root / 'label/training-latest.pt')
                        torch.save({'step': self_step}, root / 'label' / f'model-{self_step}.pt')
                def wait(self):
                    return 0

            args = argparse.Namespace(root=str(root), cache=str(root), worker='label', start_step=500000,
                                      target=550000, evaluation_samples=50000, classifier=str(root / 'classifier.pt'))
            with patch.object(runner.subprocess, 'Popen', Child), patch.object(
                runner.shutil, 'disk_usage', return_value=argparse.Namespace(free=100 * 1024**3)
            ):
                runner.worker(args)
            self.assertEqual(len(commands), 2)
            self.assertIn('strong_rf.train', commands[0])
            self.assertEqual(commands[0][commands[0].index('--steps')+1], '550000')
            self.assertIn('strong_rf.evaluate', commands[1])
            self.assertTrue(commands[1][commands[1].index('--checkpoint')+1].endswith('model-550000.pt'))
            self.assertEqual(json.loads((root / 'label/worker.json').read_text())['phase'], 'complete')

    def test_two_group_report_has_no_unstarted_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ('label', 'gaussian'):
                folder = root / 'evaluations' / f'{kind}-500000-50k'
                folder.mkdir(parents=True)
                metrics = dict(kind=kind, checkpoint_step=500000, generated_images=50000,
                               fid_train_50k_reference=3., inception_score_mean=9., generated_label_accuracy=.9)
                if kind == 'gaussian':
                    metrics['reverse_accuracy'] = .88
                (folder / 'metrics.json').write_text(json.dumps(metrics))
                Image.new('RGB', (32, 32)).save(folder / 'generated-by-class.png')
            runner.report(root, True, target=1000000, kinds=['label', 'gaussian'], start_step=500000)
            status = (root / 'STATUS.md').read_text()
            self.assertIn('1000k', status)
            self.assertNotIn('gaussian_ot', status)
            with Image.open(root / 'analysis/generation-500k-fid50k.png') as grid:
                self.assertEqual(grid.size, (1360, 715))


if __name__ == '__main__':
    unittest.main()
