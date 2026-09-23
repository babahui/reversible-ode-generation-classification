"""Sequential, restartable single-GPU training and full evaluation stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import sha256


def milestones(total, interval):
    return sorted(set(list(range(interval,total+1,interval)) + [total]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='./runs/cifar100-dct-fm')
    parser.add_argument('--data-root',default='./data')
    parser.add_argument('--steps',type=int,default=500000)
    parser.add_argument('--eval-every',type=int,default=50000)
    args = parser.parse_args()
    if min(args.steps,args.eval_every)<1:
        raise ValueError('steps and interval must be positive')
    root = Path(args.output).resolve()
    training = root/'training'
    for target in milestones(args.steps,args.eval_every):
        checkpoint = training/f'model-{target}.pt'
        if not checkpoint.exists():
            command = [sys.executable,'-m','cifar100_fm.train','--output',str(training),
                       '--data-root',args.data_root,'--steps',str(target),
                       '--keep-previous-checkpoint']
            if (training/'training-latest.pt').exists():
                command.append('--resume')
            subprocess.run(command,check=True)
        if not checkpoint.exists():
            raise RuntimeError(f'{checkpoint} missing; training may already be beyond this milestone')
        evaluation = root/'evaluations'/f'gaussian-{target}-50k'
        if (evaluation/'metrics.json').exists():
            protocol = json.loads((evaluation/'protocol.json').read_text())
            metrics = json.loads((evaluation/'metrics.json').read_text())
            if (protocol.get('checkpoint_sha256') != sha256(checkpoint)
                    or protocol.get('method') != 'heun' or protocol.get('steps') != 60
                    or protocol.get('batch_size') != 64 or protocol.get('seed') != 20260912
                    or protocol.get('dataset') != 'cifar100'
                    or metrics.get('checkpoint_step') != target
                    or metrics.get('generated_images') != 50000
                    or metrics.get('reverse_images') != 10000):
                raise RuntimeError('existing evaluation does not match the study protocol')
            print(f'verified completed stage {target}',flush=True)
            continue
        subprocess.run([sys.executable,'-m','cifar100_fm.evaluate',
            '--checkpoint',str(checkpoint),'--output',str(evaluation),
            '--data-root',args.data_root,'--cache',str(root/'reference-cache'),
            '--samples','50000','--reverse-images','10000','--batch-size','64',
            '--method','heun','--steps','60'],check=True)


if __name__ == '__main__':
    main()
