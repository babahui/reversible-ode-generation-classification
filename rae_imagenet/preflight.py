"""Read-only dataset/weight checks and zero-allocation large-model parameter count."""
import argparse
import json
from .config import DEFAULT_CONFIG, load_config
from .data import ImageNetImages
from .model import build_model
from .rae import verify_weights
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--rae-repo', default='third_party/RAEv2')
    p.add_argument('--data-root')
    p.add_argument('--weights')
    p.add_argument('--allow-subset', action='store_true')
    args = p.parse_args()
    config = load_config(args.config)
    with torch.device('meta'):
        model = build_model(config['model'], args.rae_repo)
    n = sum(p.numel() for p in model.parameters())
    report = dict(parameters=n, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  fp32_full_checkpoint_gib=16*n/2**30, fp32_train_state_with_gpu_ema_gib=20*n/2**30,
                  note='State estimates exclude activations, BF16 caches and CUDA workspaces; not a GPU benchmark')
    if args.data_root:
        train = ImageNetImages(args.data_root, 'train', allow_subset=args.allow_subset)
        val = ImageNetImages(args.data_root, 'val', classes=train.classes, allow_subset=args.allow_subset)
        report.update(train=train.signature(), validation=val.signature())
    if args.weights:
        report['verified_weights'] = verify_weights(args.weights)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
