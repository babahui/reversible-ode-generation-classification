"""Resumable RF sampling and two independent Inception FID implementations."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric
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
def extract(images, extractor, clean_extractor=None, batch_size=64):
    chunks, logits, clean = [], [], []
    for batch in torch.as_tensor(images).split(batch_size):
        batch = batch.cuda()
        f, l = extractor(batch)
        chunks.append(f.cpu().numpy())
        logits.append(l.cpu().numpy())
        if clean_extractor is not None:
            clean.append(clean_extractor(batch.float()).cpu().numpy())
    result = {'features': np.concatenate(chunks), 'logits': np.concatenate(logits)}
    if clean:
        result['clean_features'] = np.concatenate(clean)
    return result


def reference(data_root, cache, extractor):
    cache.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ['train', 'test']:
        path = cache / ('cifar10_' + split + '_torchfidelity_fp32_v1.npz')
        if not path.exists():
            dataset = CIFAR10(data_root, train=split == 'train', download=False)
            images = dataset.data
            labels = np.asarray(dataset.targets)
            if split == 'test':
                chosen = np.sort(np.concatenate([np.flatnonzero(labels == c)[:500] for c in range(10)]))
                images, labels = images[chosen], labels[chosen]
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
    p.add_argument('--data-root', default='/root/autodl-tmp/all_datasets')
    p.add_argument('--cache', required=True)
    p.add_argument('--samples', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--method', choices=['dopri5','heun','euler'], default='heun')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--cross-check', action='store_true')
    p.add_argument('--reverse-images', type=int, default=0)
    p.add_argument('--classifier', default='runs/reference-cifar10-classifier/cifar10-classifier-best.pt')
    args = p.parse_args()
    for key in ['checkpoint','output','data_root','cache','classifier']:
        setattr(args,key,str(Path(getattr(args,key)).resolve()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(args.checkpoint)
    protocol = dict(vars(args), checkpoint_sha256=fingerprint,
                    quantization='(x*127.5+128).clamp(0,255).uint8',
                    reference='all 50000 original train images; legacy comparison: first 500 per class of test',
                    feature_extractor='torch-fidelity inception-v3-compat 2048, FP32')
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        for key in ['checkpoint','output','data_root','cache','classifier']:
            if key in old:old[key]=str(Path(old[key]).resolve())
        # samples may increase to reuse all stored batches, everything else fixed.
        if any(old.get(k) != v for k,v in protocol.items() if k != 'samples'):
            raise RuntimeError('output belongs to another protocol/checkpoint')
    write_json(protocol_path, protocol)
    model, config, prior, checkpoint_step = load_model(args.checkpoint)
    conditional = config['kind'] in {'label','label_ot'}
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048','logits_unbiased']).eval().cuda()
    clean_extractor = None
    official_stats = None
    if args.cross_check:
        from cleanfid.features import build_feature_extractor, get_reference_statistics
        clean_extractor = build_feature_extractor('legacy_tensorflow', use_dataparallel=False)
        mu, sigma = get_reference_statistics('cifar10',32,mode='legacy_tensorflow',split='train')
        official_stats = {'mu':mu, 'sigma':sigma}
    refs = reference(args.data_root, Path(args.cache), extractor)
    batches = out/'batches'
    batches.mkdir(exist_ok=True)
    start = time.monotonic()
    classifier = None
    if config['kind'] != 'unconditional' and Path(args.classifier).exists():
        from unified_transport.reference_classifier import load_reference_classifier
        classifier, _ = load_reference_classifier(args.classifier, torch.device('cuda'))
    for offset in range(0, args.samples, args.batch_size):
        path = batches/f'{offset:07d}.npz'
        count = min(args.batch_size, args.samples-offset)
        if path.exists() and len(np.load(path)['labels']) == count:
            continue
        labels = torch.arange(offset,offset+count,device='cuda') % 10
        generator = torch.Generator(device='cuda').manual_seed(args.seed+offset)
        initial = initial_noise(labels, config, prior, generator)
        tic = time.monotonic()
        with torch.inference_mode():
            generated, nfe = integrate(model, initial, labels if conditional else None,
                args.method,args.steps,tol=args.tol)
            if not bool(generated.isfinite().all()):
                raise RuntimeError('nonfinite generated images')
            images = to_uint8(generated)
            result = extract(images,extractor,clean_extractor)
            result.update(images=images.cpu().numpy(),labels=labels.cpu().numpy(),
                          nfe=np.asarray(nfe),seconds=np.asarray(time.monotonic()-tic))
            if classifier is not None:
                result['predicted_labels'] = classifier(images.float()/127.5-1).argmax(1).cpu().numpy()
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
    legacy_indices=np.arange(min(2560,len(features)))
    metrics={'checkpoint_step':checkpoint_step,'kind':config['kind'],'generated_images':len(features),
             'fid_train_50k_reference':fid(features,refs['train']),
             'fid_test_5k_reference':fid(features,refs['test']),
             'legacy_2560_vs_test5k_fid':fid(features[legacy_indices],refs['test']),
             'mean_nfe':float(np.mean([r['nfe'] for r in records])),
             'generation_seconds':float(sum(r['seconds'] for r in records)),
             'model_parameters':sum(p.numel() for p in model.parameters())}
    metrics.update(isc_features_to_metric(logits))
    metrics['kid_legacy_2560_vs_test5k']=kid_features_to_metric(
        torch.from_numpy(features[legacy_indices]),torch.from_numpy(refs['test']),
        kid_subset_size=min(1000,len(legacy_indices)),kid_subsets=100,verbose=False,rng_seed=2020)
    images=np.concatenate([r['images'] for r in records])
    dark=(images<=5).all(1)
    border=np.ones((32,32),dtype=bool);border[4:-4,4:-4]=False
    metrics['near_black_border_fraction']=float(dark[:,border].mean())
    if classifier is not None:
        predictions=np.concatenate([r['predicted_labels'] for r in records])
        metrics['generated_label_accuracy']=float((predictions==labels).mean())
        confusion=np.zeros((10,10),dtype=np.int64)
        np.add.at(confusion,(labels,predictions),1)
        metrics['generated_confusion_matrix']=confusion.tolist()
        metrics['generated_accuracy_per_class']=(confusion.diagonal()/confusion.sum(1).clip(1)).tolist()
    if args.cross_check:
        clean=np.concatenate([r['clean_features'] for r in records])
        metrics['official_cleanfid_legacy_tensorflow_train']=fid(clean,official_stats)
        metrics['cross_extractor_feature_mae']=float(np.abs(clean-features).mean())
        metrics['fid_cross_implementation_absolute_difference']=abs(metrics['fid_train_50k_reference']-metrics['official_cleanfid_legacy_tensorflow_train'])
    if prior is not None and args.reverse_images:
        dataset=CIFAR10(args.data_root,train=False)
        correct=0;count=0
        confusion=np.zeros((10,10),dtype=np.int64)
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
                print(json.dumps({'reverse_evaluated':count,'correct':correct}),flush=True)
        metrics['reverse_accuracy']=correct/count;metrics['reverse_images']=count
        metrics['reverse_confusion_matrix']=confusion.tolist()
        metrics['reverse_accuracy_per_class']=(confusion.diagonal()/confusion.sum(1).clip(1)).tolist()
    if config['kind']!='unconditional':
        selected=np.concatenate([np.flatnonzero(labels==c)[:10] for c in range(10)])
        save_image(torch.from_numpy(images[selected]).float()/255,out/'generated-by-class.png',nrow=10)
    write_json(out/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
