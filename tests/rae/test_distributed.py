"""Two CPU/Gloo ranks exercise DDP accumulation and per-rank RNG checkpointing."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from rae_imagenet.config import load_config
from rae_imagenet.state import EMA, restore_rng
from rae_imagenet.train import update, save_checkpoint


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(2,2,1)
        self.embedding=torch.nn.Embedding(1001,2)
    def forward(self,x,t,labels):
        return self.conv(x)+self.embedding(labels)[:,:,None,None]+t[:,None,None,None]


def worker(rank,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=2)
    try:
        torch.manual_seed(17)
        model=Toy();wrapped=DistributedDataParallel(model)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        config=load_config();tc=config['train'];tc.update(global_batch=4,microbatch=1)
        ema=EMA(model,.999) if rank==0 else None
        torch.manual_seed(123+rank)
        batches=[(torch.randn(1,2,4,4),torch.tensor([rank])) for _ in range(2)]
        def one():
            result=update(wrapped,lambda x:x,iter(batches),opt,tc,1000,torch.device('cpu'),2)
            if ema is not None:ema.update(model)
            return result
        one()
        save_checkpoint(model,ema,opt,config,{'classes':['n01440764','n01443537']},1,1.,
                        SimpleNamespace(output=folder),rank,2,torch.device('cpu'))
        loss=one()[0];expected=copy.deepcopy(model.state_dict())
        saved=torch.load(Path(folder)/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert saved['world_size']==2 and len(saved['rng_by_rank'])==2
        assert not torch.equal(saved['rng_by_rank'][0]['torch'],saved['rng_by_rank'][1]['torch'])
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        if ema is not None:ema.load(saved['ema'])
        restore_rng(saved['rng_by_rank'][rank])
        repeated=one()[0]
        assert torch.equal(loss,repeated)
        for key,value in model.state_dict().items():assert torch.equal(value,expected[key]),key
        flat=torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered=[torch.zeros_like(flat) for _ in range(2)]
        dist.all_gather(gathered,flat)
        assert torch.equal(gathered[0],gathered[1])
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_two_rank_accumulation_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker,args=(folder,),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
