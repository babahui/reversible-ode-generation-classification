"""CPU-only plots and matched-checkpoint comparisons, regenerated after evaluations."""
import csv
from pathlib import Path


def summarize(root, rows, kinds=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from .run_five_gpu import KINDS
    KINDS = list(KINDS if kinds is None else kinds)
    pairs = [('gaussian','gaussian_ot'), ('gaussian','gaussian_ce'),
             ('gaussian_ot','gaussian_ot_ce'), ('gaussian_ce','gaussian_ot_ce')]
    pairs = [(a, b) for a, b in pairs if a in KINDS and b in KINDS]
    if KINDS == ['label', 'gaussian']:
        pairs = [('label', 'gaussian')]
    root = Path(root)
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    columns = ['kind', 'checkpoint_step', 'generated_images', 'fid_train_50k_reference',
               'inception_score_mean', 'generated_label_accuracy', 'reverse_accuracy']
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for kind in KINDS:
        points = sorted([m for m in rows if m['kind'] == kind and m['generated_images'] == 5000],
                        key=lambda m: m['checkpoint_step'])
        for ax, key, title in zip(axes, ['fid_train_50k_reference', 'generated_label_accuracy', 'reverse_accuracy'],
                                  ['FID (5k generated / train50k)', 'Generated class accuracy', 'Reverse accuracy (test10k)']):
            usable = [m for m in points if key in m]
            ax.plot([m['checkpoint_step']/1000 for m in usable], [m[key] for m in usable], 'o-', label=kind)
            ax.set_title(title); ax.set_xlabel('Training step (k)'); ax.grid(alpha=.25)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / 'trends-5k.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for kind in KINDS:
        points = sorted([m for m in rows if m['kind']==kind and m['generated_images']==50000],key=lambda m:m['checkpoint_step'])
        for ax,key,title in zip(axes,['fid_train_50k_reference','generated_label_accuracy','reverse_accuracy'],
                               ['FID (50k generated / train50k)','Generated class accuracy','Reverse accuracy (test10k)']):
            usable=[m for m in points if key in m]
            ax.plot([m['checkpoint_step']/1000 for m in usable],[m[key] for m in usable],'o-',label=kind)
            ax.set_title(title);ax.set_xlabel('Training step (k)');ax.grid(alpha=.25)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'trends-50k.png',dpi=160);plt.close(fig)
    text = ['# %d 组同 step 分析' % len(KINDS), '',
            '后续统一使用FID50k：50,000生成 / 原始train50k，EMA、Heun60/NFE120、batch64、seed20260912、FP32；与Label100k=4.4815相同设置。',
            '当前训练段保持运行，结束后评估；增加采样量会延长评估阶段。下列比较均固定训练step和生成数；单seed，无显著性结论。', '',
            '## 正式 FID50k 已完成结果', '',
            '| 方法 | 训练step | FID50k | IS | 生成类别准确率 | 反向分类 |', '|---|---:|---:|---:|---:|---:|']
    full=[m for m in rows if m['generated_images']==50000]
    for m in sorted(full,key=lambda m:(m['checkpoint_step'],KINDS.index(m['kind']))):
        reverse='%.2f%%'%(100*m['reverse_accuracy']) if 'reverse_accuracy' in m else '—'
        text.append('| %s | %d | %.4f | %.3f | %.2f%% | %s |'%(m['kind'],m['checkpoint_step'],m['fid_train_50k_reference'],m['inception_score_mean'],100*m['generated_label_accuracy'],reverse))
    text+=['','![正式FID50k趋势](trends-50k.png)','']
    for step in sorted({m['checkpoint_step'] for m in full}):
        group={m['kind']:m for m in full if m['checkpoint_step']==step}
        if not all(kind in group for kind in KINDS):continue
        text+=['### %dk 正式对比'%(step//1000),'','| 对照 | FID变化 | 反向分类变化（百分点） |','|---|---:|---:|']
        for a,b in pairs:
            reverse = ('%+.2f' % (100*(group[b]['reverse_accuracy']-group[a]['reverse_accuracy']))
                       if 'reverse_accuracy' in group[a] and 'reverse_accuracy' in group[b] else '—')
            text.append('| %s → %s | %+.4f | %s |'%(a,b,group[b]['fid_train_50k_reference']-group[a]['fid_train_50k_reference'],reverse))
        canvas=Image.new('RGB',(680*len(KINDS),715),'white')
        for i,kind in enumerate(KINDS):
            grid=Image.open(root/group[kind]['evaluation']/'generated-by-class.png').convert('RGB')
            grid=grid.resize((680,680),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST)
            canvas.paste(grid,(680*i,35));ImageDraw.Draw(canvas).text((680*i+10,10),'%s @ %dk / FID50k'%(kind,step//1000),fill='black')
        name='generation-%dk-fid50k.png'%(step//1000);canvas.save(out/name)
        text+=['','![正式固定样例](%s)'%name,'']
    text+=['## 历史 FID5k（不与上述FID50k混比）','']
    complete = []
    for step in sorted({m['checkpoint_step'] for m in rows}):
        group = {m['kind']: m for m in rows if m['checkpoint_step'] == step and m['generated_images'] == 5000}
        if not all(kind in group for kind in KINDS):
            continue
        complete.append(step)
        text += ['## %dk：5k 生成，完整测试集反向分类' % (step//1000), '',
                 '| 对照 | FID变化（后者−前者） | 反向分类变化（百分点） |', '|---|---:|---:|']
        for a, b in pairs:
            reverse = ('%+.2f' % (100*(group[b]['reverse_accuracy']-group[a]['reverse_accuracy']))
                       if 'reverse_accuracy' in group[a] and 'reverse_accuracy' in group[b] else '—')
            text.append('| %s → %s | %+.4f | %s |' % (a, b,
                group[b]['fid_train_50k_reference']-group[a]['fid_train_50k_reference'], reverse))
        if 'gaussian_ot_ce' in group and 'label' in group:
            gap = group['gaussian_ot_ce']['fid_train_50k_reference']/group['label']['fid_train_50k_reference']-1
            text += ['', 'Gaussian+OT+CE 相对同 step Label 的 FID5k 差距：%+.1f%%；正式保真目标用 FID50k 判定。' % (100*gap), '']
        tiles = []
        for kind in KINDS:
            grid = Image.open(root / group[kind]['evaluation'] / 'generated-by-class.png').convert('RGB')
            grid = grid.resize((680,680), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
            tile = Image.new('RGB',(680,715),'white'); tile.paste(grid,(0,35))
            ImageDraw.Draw(tile).text((10,10), '%s @ %dk' % (kind, step//1000), fill='black')
            tiles.append(tile)
        canvas = Image.new('RGB',(680*len(KINDS),715),'white')
        for i,tile in enumerate(tiles): canvas.paste(tile,(680*i,0))
        canvas.save(out / ('generation-%dk.png' % (step//1000)))
        text += ['![固定顺序样例](generation-%dk.png)' % (step//1000), '']
    if not complete:
        text += ['尚无上述方法共同的5k生成评估；正式50k生成结果见上表。']
    (out / 'REPORT.md').write_text('\n'.join(text)+'\n')
