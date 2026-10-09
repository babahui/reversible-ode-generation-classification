"""Check matching frozen RAE weights on real images before DDT training."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from .data import ImageNetImages
from .rae import Encoder, Decoder, verify_weights
from .state import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--rae-repo',default='third_party/RAEv2')
    p.add_argument('--dino-repo',default='third_party/dinov3')
    p.add_argument('--device',default='cuda')
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--allow-subset',action='store_true')
    a=p.parse_args()
    if a.count<1:p.error('count must be positive')
    torch.set_num_threads(4)
    verify_weights(a.weights)
    device=torch.device(a.device)
    data=ImageNetImages(a.data_root,'val',allow_subset=a.allow_subset)
    encoder=Encoder(a.rae_repo,a.dino_repo,a.weights,device)
    decoder=Decoder(a.rae_repo,a.weights,device)
    indices=torch.linspace(0,len(data)-1,min(a.count,len(data))).long().tolist()
    tiles=[];psnr=[]
    for index in indices:
        image,_=data[index];image=image.unsqueeze(0).to(device)
        latent=encoder(image);rec=decoder(latent)
        if not torch.isfinite(latent).all() or not torch.isfinite(rec).all():raise FloatingPointError('Nonfinite reconstruction')
        psnr.append(float(-10*(rec-image).square().mean().clamp_min(1e-12).log10()))
        tiles.extend([image.cpu(),rec.cpu()])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    save_image(torch.cat(tiles),out/'reconstructions.png',nrow=8)
    report=dict(count=len(indices),mean_psnr=sum(psnr)/len(psnr),per_image_psnr=psnr,indices=indices,
                latent_shape=list(latent.shape[1:]),subset_check_only=a.allow_subset)
    write_json(out/'reconstruction.json',report)
    print(report)


if __name__=='__main__':main()
