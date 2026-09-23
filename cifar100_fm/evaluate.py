"""Original CIFAR-100 FID/IS and label-free reverse ODE evaluation protocol."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR100
from torchvision.utils import save_image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_features_to_metric

from .common import load_model, initial_noise, integrate, to_uint8, write_json, sha256


def fid(a, b):
    sa = fid_features_to_statistics(torch.from_numpy(a))
    sb = b if isinstance(b, dict) else fid_features_to_statistics(torch.from_numpy(b))
    # Reference statistics distributed by clean-fid use float64 means.
    sa['mu'] = sa['mu'].astype(np.float64)
    sb = {k: v.astype(np.float64) for k, v in sb.items()}
    return fid_statistics_to_metric(sa, sb, False)['frechet_inception_distance']


def atomic_npz(path, **arrays):
    tmp = Path(str(path) + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


@torch.inference_mode()
def extract(images, extractor, batch_size=64):
    chunks, logits = [], []
    for batch in torch.as_tensor(images).split(batch_size):
        batch = batch.cuda()
        f, l = extractor(batch)
        chunks.append(f.cpu().numpy())
        logits.append(l.cpu().numpy())
    result = {'features': np.concatenate(chunks), 'logits': np.concatenate(logits)}
    return result


def reference(data_root, cache, extractor):
    cache.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ['train', 'test']:
        path = cache / ('cifar100_' + split + '_torchfidelity_fp32_v1.npz')
        if not path.exists():
            dataset = CIFAR100(data_root, train=split == 'train', download=False)
            images = dataset.data
            labels = np.asarray(dataset.targets)
            collected = []
            for offset in range(0, len(images), 1024):
                x = torch.from_numpy(images[offset:offset+1024]).permute(0,3,1,2).contiguous()
                collected.append(extract(x, extractor)['features'])
                print(json.dumps({'reference': split, 'done': min(offset+1024,len(images)), 'total':len(images)}), flush=True)
            atomic_npz(path, features=np.concatenate(collected), labels=labels)
        outputs[split] = np.load(path)['features']
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--cache', required=True)
    p.add_argument('--samples', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--method', choices=['dopri5','heun','euler'], default='heun')
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--reverse-images', type=int, default=10000)
    args = p.parse_args()
    if min(args.samples,args.batch_size,args.steps)<1 or args.reverse_images<0:
        raise ValueError('invalid evaluation sizes')
    for key in ['checkpoint','output','data_root','cache']:
        setattr(args,key,str(Path(getattr(args,key)).resolve()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(args.checkpoint)
    protocol = dict(vars(args), checkpoint_sha256=fingerprint, dataset='cifar100',num_classes=100,
                    quantization='(x*127.5+128).clamp(0,255).uint8',
                    reference='all 50000 original CIFAR-100 train images; optional test reference uses all 10000',
                    feature_extractor='torch-fidelity inception-v3-compat 2048, FP32')
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        for key in ['checkpoint','output','data_root','cache']:
            if key in old:old[key]=str(Path(old[key]).resolve())
        # samples may increase to reuse all stored batches, everything else fixed.
        if any(old.get(k) != v for k,v in protocol.items() if k != 'samples'):
            raise RuntimeError('output belongs to another protocol/checkpoint')
    write_json(protocol_path, protocol)
    model, config, prior, checkpoint_step = load_model(args.checkpoint)
    if config.get('dataset') != 'cifar100' or config.get('num_classes') != 100:
        raise ValueError('requires a CIFAR-100 checkpoint')
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048','logits_unbiased']).eval().cuda()
    refs = reference(args.data_root, Path(args.cache), extractor)
    batches = out/'batches'
    batches.mkdir(exist_ok=True)
    start = time.monotonic()
    for offset in range(0, args.samples, args.batch_size):
        path = batches/f'{offset:07d}.npz'
        count = min(args.batch_size, args.samples-offset)
        if path.exists() and len(np.load(path)['labels']) == count:
            continue
        labels = torch.arange(offset,offset+count,device='cuda') % 100
        generator = torch.Generator(device='cuda').manual_seed(args.seed+offset)
        initial = initial_noise(labels, config, prior, generator)
        tic = time.monotonic()
        with torch.inference_mode():
            generated, nfe = integrate(model, initial, None,
                args.method,args.steps,tol=args.tol)
            if not bool(generated.isfinite().all()):
                raise RuntimeError('nonfinite generated images')
            images = to_uint8(generated)
            result = extract(images,extractor)
            result.update(images=images.cpu().numpy(),labels=labels.cpu().numpy(),
                          nfe=np.asarray(nfe),seconds=np.asarray(time.monotonic()-tic))
        atomic_npz(path,**result)
        write_json(out/'progress.json',{'generated':offset+count,'target':args.samples,
                  'elapsed_seconds_this_invocation':time.monotonic()-start,'last_nfe':nfe})
        print(json.dumps({'generated':offset+count,'target':args.samples,'nfe':nfe,'seconds':time.monotonic()-tic}),flush=True)
        if offset == 0:
            save_image(images.float()/255, out/'first-batch.png',nrow=8)
    records=[np.load(batches/f'{i:07d}.npz') for i in range(0,args.samples,args.batch_size)]
    features=np.concatenate([r['features'] for r in records])
    labels=np.concatenate([r['labels'] for r in records])
    logits=torch.from_numpy(np.concatenate([r['logits'] for r in records]))
    metrics={'checkpoint_step':checkpoint_step,'kind':config['kind'],'generated_images':len(features),
             'dataset':'cifar100','num_classes':100,'time_encoding':'scalar',
             'prior_version':config['prior_version'],
             'generated_label_accuracy':None,
             'generated_label_accuracy_note':'未配置独立CIFAR-100评分分类器；不使用CIFAR-10评分器。',
             'generated_count_per_class':np.bincount(labels,minlength=100).tolist(),
             'fid_train_50k_reference':fid(features,refs['train']),
             'fid_test_10k_reference':fid(features,refs['test']),
             'mean_nfe':float(np.mean([r['nfe'] for r in records])),
             'generation_seconds':float(sum(r['seconds'] for r in records)),
             'model_parameters':sum(p.numel() for p in model.parameters())}
    metrics.update(isc_features_to_metric(logits))
    images=np.concatenate([r['images'] for r in records])
    dark=(images<=5).all(1)
    border=np.ones((32,32),dtype=bool);border[4:-4,4:-4]=False
    metrics['near_black_border_fraction']=float(dark[:,border].mean())
    if prior is not None and args.reverse_images:
        dataset=CIFAR100(args.data_root,train=False)
        correct=0;count=0
        confusion=np.zeros((100,100),dtype=np.int64)
        with torch.inference_mode():
            for offset in range(0,min(args.reverse_images,len(dataset)),args.batch_size):
                n=min(args.batch_size,args.reverse_images-offset,len(dataset)-offset)
                x=torch.from_numpy(dataset.data[offset:offset+n]).permute(0,3,1,2).cuda().float()/127.5-1
                # Crucially: no ground truth labels are passed into this reverse solve.
                z,_=integrate(model,x,None,args.method,args.steps,reverse=True,tol=args.tol)
                predicted=prior.classify(z).cpu().numpy()
                truth=np.asarray(dataset.targets[offset:offset+n])
                np.add.at(confusion,(truth,predicted),1)
                correct+=int((predicted==truth).sum());count+=n
                write_json(out/'reverse-progress.json',{'evaluated':count,'target':min(args.reverse_images,len(dataset)), 'correct':correct})
                print(json.dumps({'reverse_evaluated':count,'correct':correct}),flush=True)
        metrics['reverse_accuracy']=correct/count;metrics['reverse_images']=count
        metrics['reverse_confusion_matrix']=confusion.tolist()
        metrics['reverse_accuracy_per_class']=(confusion.diagonal()/confusion.sum(1).clip(1)).tolist()
        metrics['class_names']=dataset.classes
        metrics['center_modes_rgb_uv']=prior.modes
    if config['kind']!='unconditional':
        selected=np.concatenate([np.flatnonzero(labels==c)[:10] for c in range(100)])
        save_image(torch.from_numpy(images[selected]).float()/255,out/'generated-by-class.png',nrow=10)
        for first in range(0,100,10):
            chosen=np.concatenate([np.flatnonzero(labels==c)[:10] for c in range(first,first+10)])
            if len(chosen):
                save_image(torch.from_numpy(images[chosen]).float()/255,out/f'classes-{first:02d}-{first+9:02d}.png',nrow=10)
    write_json(out/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
