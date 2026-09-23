"""Compare completed, protocol-matched evaluations without inventing missing rows."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def collect(root):
    records={}
    for path in Path(root).glob('evaluations/*/metrics.json'):
        metric=json.loads(path.read_text())
        protocol=json.loads((path.parent/'protocol.json').read_text())
        if metric.get('dataset')!='cifar100':
            raise ValueError('CIFAR-100 results only')
        if metric.get('reverse_images')!=10000 or metric.get('generated_images')!=50000:
            continue
        step=metric['checkpoint_step']
        if step in records:
            raise ValueError(f'ambiguous duplicate evaluation step {step} under {root}')
        records[step]=(metric,protocol,path.parent)
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--dual',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    a,b=collect(args.baseline),collect(args.dual)
    steps=sorted(a.keys() & b.keys())
    if not steps:
        raise RuntimeError('no shared completed 50K generation / 10K classification steps yet')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    markdown=['# CIFAR-100 matched-step comparison','',
              '| Total step | Method | Reverse accuracy | FID train50K | IS |',
              '|---|---|---:|---:|---:|']
    for step in steps:
        for key in ['method','steps','seed','batch_size','samples','reverse_images','quantization',
                    'reference','feature_extractor','dataset','num_classes','tol']:
            if a[step][1].get(key)!=b[step][1].get(key):
                raise ValueError(f'protocol mismatch at {step}: {key}')
        if a[step][0]['kind']!='gaussian' or b[step][0]['kind']!='gaussian_dual_pad4_js':
            raise ValueError('baseline must be pure FM and dual must be CE+JS')
        for name,record in [('FM',a[step]),('Dual-pad4 CE+JS',b[step])]:
            m=record[0]
            rows.append(dict(step=step,method=name,reverse_accuracy=m['reverse_accuracy'],
                fid_train_50k=m['fid_train_50k_reference'],is_mean=m['inception_score_mean']))
            markdown.append(f"| {step} | {name} | {100*m['reverse_accuracy']:.2f}% | "
                f"{m['fid_train_50k_reference']:.3f} | {m['inception_score_mean']:.3f} |")
        for first in range(0,100,10):
            filename=f'classes-{first:02d}-{first+9:02d}.png'
            paths=[record[2]/filename for record in (a[step],b[step])]
            if not all(p.exists() for p in paths):
                continue
            with Image.open(paths[0]) as left,Image.open(paths[1]) as right:
                canvas=Image.new('RGB',(left.width+right.width+12,max(left.height,right.height)+24),'white')
                canvas.paste(left.convert('RGB'),(0,24));canvas.paste(right.convert('RGB'),(left.width+12,24))
                draw=ImageDraw.Draw(canvas)
                draw.text((3,5),f'FM step {step}',fill='black')
                draw.text((left.width+15,5),'Dual-pad4 CE+JS',fill='black')
                canvas.save(out/f'{step}-{filename}')
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    markdown+=['','Only completed, matched-total-step evaluations are included.',
               'Generated-label accuracy is not measured without an independent CIFAR-100 scorer.']
    (out/'README.md').write_text('\n'.join(markdown)+'\n')
    print(f'Compared {len(steps)} matched checkpoints; output: {out}')


if __name__=='__main__':
    main()
