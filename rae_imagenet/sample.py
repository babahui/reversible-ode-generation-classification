"""Generate images from a conditional FM EMA checkpoint without a dataset."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from .config import LATENT_SHAPE
from .flow import integrate, indexed_noise
from .model import build_model
from .rae import Decoder
from .state import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--rae-repo', default='third_party/RAEv2')
    p.add_argument('--output', required=True)
    p.add_argument('--labels', default='0,1,2,3,4,5,6,7,8,9', help='Indices in checkpoint class_to_idx order')
    p.add_argument('--cfg', type=float, default=2.)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261208)
    args = p.parse_args()
    if args.batch_size < 1: p.error('batch-size must be positive')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    labels = [int(v) for v in args.labels.split(',')]
    if not labels or any(y < 0 or y >= config['model']['num_classes'] for y in labels):
        p.error('Labels must be in [0, num_classes)')
    model = build_model(config['model'], args.rae_repo, checkpointing=False)
    model.load_state_dict(saved['ema'], strict=True)
    class_names = saved['dataset']['classes']
    del saved
    model.cuda().eval()
    decoder = Decoder(args.rae_repo, args.weights, torch.device('cuda'))
    images = []
    for start in range(0, len(labels), args.batch_size):
        y = torch.tensor(labels[start:start+args.batch_size], device='cuda')
        noise = indexed_noise(range(start, start+len(y)), LATENT_SHAPE, args.seed, y.device)
        z = integrate(model, noise, y, config['model']['num_classes'], args.steps, args.cfg)
        images.append(decoder(z).cpu())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat(images), out / 'samples.png', nrow=min(10, len(labels)))
    write_json(out / 'sampling.json', dict(labels=labels, wnids=[class_names[y] if y < len(class_names) else None for y in labels],
               seed=args.seed, cfg=args.cfg, steps=args.steps, checkpoint=args.checkpoint))


if __name__ == '__main__':
    main()
