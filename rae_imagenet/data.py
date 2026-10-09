"""ImageFolder input without a dataset-sized latent cache.

Each optimizer step defines its own deterministic global image/view batch.
DataLoader prefetch and worker RNG cannot change the stream after a restart.
Sampling is uniform over images with replacement, sharded across DDP ranks.
"""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

COUNTS = {'train': 1281167, 'val': 50000}
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256), transforms.ToTensor()])


class ImageNetImages(ImageFolder):
    def __init__(self, data_root, split, classes=None, allow_subset=False):
        self.split = split
        super().__init__(str(Path(data_root) / split))
        if classes is not None and self.classes != list(classes):
            raise ValueError('Train and validation WNID class ordering must match')
        if not allow_subset and (len(self.classes) != 1000 or len(self) != COUNTS[split]):
            raise ValueError(f'Expected full ImageNet {split}: 1000 classes, {COUNTS[split]} '
                             f'images; got {len(self.classes)} classes, {len(self)} images. '
                             'Use --allow-subset only for pipeline checks.')
        if len(self.classes) > 1000:
            raise ValueError('Found more than 1000 classes')
        for name in self.classes:
            if len(name) != 9 or not name.startswith('n') or not name[1:].isdigit():
                raise ValueError('Class directories must be ImageNet WNIDs, e.g. n01440764')

    def __getitem__(self, item):
        index, flip = item if isinstance(item, tuple) else (item, False)
        path, label = self.samples[index]
        with Image.open(path) as image:
            x = TRANSFORM(image.convert('RGB'))
        return (x.flip(-1) if flip else x), label

    def signature(self):
        digest = hashlib.sha256(json.dumps(self.class_to_idx, sort_keys=True).encode())
        base = Path(self.root)
        for path, label in self.samples:
            digest.update(f'{Path(path).relative_to(base)}\t{label}\n'.encode())
        return {'count': len(self), 'classes': self.classes, 'manifest_sha256': digest.hexdigest()}


class StepBatchSampler(Sampler):
    def __init__(self, count, global_batch, microbatch, seed, start_step=0,
                 max_steps=None, rank=0, world_size=1):
        if count < 1 or global_batch % (world_size * microbatch):
            raise ValueError('Invalid dataset count or batch divisibility')
        self.count, self.global_batch, self.microbatch = count, global_batch, microbatch
        self.seed, self.start_step, self.max_steps = seed, start_step, max_steps
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        step = self.start_step
        local_batch = self.global_batch // self.world_size
        while self.max_steps is None or step < self.max_steps:
            g = torch.Generator().manual_seed((self.seed + step * 1000003) % (2**63 - 1))
            ids = torch.randint(self.count, (self.global_batch,), generator=g).tolist()
            flips = torch.randint(2, (self.global_batch,), generator=g).tolist()
            begin = self.rank * local_batch
            for offset in range(begin, begin + local_batch, self.microbatch):
                yield [(ids[i], bool(flips[i])) for i in range(offset, offset + self.microbatch)]
            step += 1
