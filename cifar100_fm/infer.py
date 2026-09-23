"""Generate a class grid or classify clean CIFAR-100 test images, without a head."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image

from .common import load_model, initial_noise, integrate, to_uint8, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['sample', 'classify'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--method', choices=['heun', 'euler'], default='heun')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--images', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260912)
    args = parser.parse_args()
    if min(args.batch_size,args.steps,args.per_class,args.images) < 1:
        raise ValueError('sizes must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, config, prior, step = load_model(args.checkpoint, args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.task == 'sample':
        labels = torch.arange(100, device=args.device).repeat_interleave(args.per_class)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        noise = initial_noise(labels, config, prior, generator)
        chunks=[]
        for z in noise.split(args.batch_size):
            result,_=integrate(model,z,None,args.method,args.steps)
            if not bool(result.isfinite().all()):
                raise RuntimeError('nonfinite generated states')
            chunks.append(to_uint8(result).cpu())
        generated = torch.cat(chunks)
        for first in range(0,100,10):
            save_image(generated[first*args.per_class:(first+10)*args.per_class].float()/255,
                       out/f'classes-{first:02d}-{first+9:02d}.png', nrow=args.per_class)
        write_json(out/'sample.json',dict(step=step,method=args.method,steps=args.steps,
                                         seed=args.seed,per_class=args.per_class))
    else:
        dataset = CIFAR100(args.data_root,train=False,download=False)
        count = min(args.images,len(dataset))
        predictions = []
        for start in range(0,count,args.batch_size):
            x = torch.from_numpy(dataset.data[start:min(start+args.batch_size,count)])
            x = x.permute(0,3,1,2).to(args.device).float()/127.5-1
            encoded,_ = integrate(model,x,None,args.method,args.steps,reverse=True)
            if not bool(encoded.isfinite().all()):
                raise RuntimeError('nonfinite reverse states')
            predictions.extend(prior.classify(encoded).cpu().tolist())
            print(f'classified {len(predictions)}/{count}',flush=True)
        truth = np.asarray(dataset.targets[:count])
        confusion = np.zeros((100,100),dtype=np.int64)
        np.add.at(confusion,(truth,predictions),1)
        write_json(out/'classification.json',dict(checkpoint_step=step,method=args.method,
            steps=args.steps,images=count,accuracy=float((truth==predictions).mean()),
            predictions=predictions,targets=truth.tolist(),confusion_matrix=confusion.tolist()))


if __name__ == '__main__':
    main()
