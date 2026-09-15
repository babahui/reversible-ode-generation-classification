"""Measure target-distribution shift from augmentation alone, using only real images."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from .evaluate import extract, fid
from .common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root);out=root/'augmentation-audit';out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dataset=CIFAR10('/root/autodl-tmp/all_datasets',train=False)
    labels=np.asarray(dataset.targets)
    chosen=np.sort(np.concatenate([np.flatnonzero(labels==c)[:500] for c in range(10)]))
    original=dataset.data[chosen]
    rng=np.random.RandomState(20260912)
    flips=rng.rand(len(original))<.5
    offsets=rng.randint(0,9,(len(original),2))
    flipped=original.copy();flipped[flips]=flipped[flips,:,::-1,:]
    padded=np.pad(original,((0,0),(4,4),(4,4),(0,0)),constant_values=0)
    cropped=np.stack([padded[i,dy:dy+32,dx:dx+32] for i,(dy,dx) in enumerate(offsets)])
    cropped[flips]=cropped[flips,:,::-1,:]
    train=np.load(root/'reference-cache/cifar10_train_torchfidelity_fp32_v1.npz')['features']
    test=np.load(root/'reference-cache/cifar10_test_torchfidelity_fp32_v1.npz')['features']
    extractor=FeatureExtractorInceptionV3('inception-v3-compat',['2048','logits_unbiased']).eval().cuda()
    result={'protocol':'same 5000 class-balanced real test images; compare to 50000 unaugmented train images; FP32 torch-fidelity; augmentation seed 20260912',
            'unaugmented_real':{'fid_train_reference':fid(test,train)}}
    for name,images in [('flip_only',flipped),('black_padding_crop_and_flip',cropped)]:
        features=extract(torch.from_numpy(images).permute(0,3,1,2).contiguous(),extractor)['features']
        border=np.ones((32,32),bool);border[4:-4,4:-4]=False
        result[name]={'fid_train_reference':fid(features,train),
                      'near_black_border_fraction':float((images<=5).all(-1)[:,border].mean())}
        print(name,result[name],flush=True)
    selected=np.arange(32)
    montage=np.concatenate([original[selected],flipped[selected],cropped[selected]])
    save_image(torch.from_numpy(montage).permute(0,3,1,2).float()/255,out/'real-augmentation-grid.png',nrow=8)
    result['interpretation']='This is a controlled augmentation-only distribution comparison, not an additive FID correction or a retrained generator ablation.'
    write_json(out/'metrics.json',result)


if __name__=='__main__':main()
