import datetime
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def tensor_bytes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = tensor_bytes(obj) + 512 * 1024**2
    if shutil.disk_usage(path.parent).free < needed:
        raise OSError(f'Insufficient free space for atomic checkpoint: need approximately {needed / 2**30:.2f} GiB')
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


class EMA:
    def __init__(self, model, decay, device='cpu'):
        self.decay = decay
        self.state = {k: v.detach().to(device).clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            target = self.state[key]
            value = value.detach().to(target.device)
            if target.is_floating_point():
                target.lerp_(value, 1 - self.decay)
            else:
                target.copy_(value)

    def load(self, state):
        if self.state.keys() != state.keys():
            raise ValueError('EMA keys do not match model')
        for key, value in state.items():
            self.state[key].copy_(value.to(self.state[key].device))
