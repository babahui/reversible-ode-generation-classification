"""Finite, resumable study with explicit quality gates and continuously written status."""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import write_json


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Study:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.repo=Path(__file__).resolve().parents[1]
        self.root.mkdir(parents=True,exist_ok=True)
        self.state={'started_or_resumed':now(),'pid':os.getpid(),'phase':'initializing',
                    'objective':'强RF生成基线；同主干低频Gaussian+OT+CE统一正向生成和反向分类；FID退化暂定≤10%',
                    'results':{}}
        old=self.root/'study.json'
        if old.exists():self.state['results']=json.loads(old.read_text()).get('results',{})
        self.env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(self.repo))

    def status(self,phase,**details):
        self.state.update(phase=phase,updated=now(),**details)
        write_json(self.root/'study.json',self.state)
        lines=['# CIFAR-10 强 RF 基线与统一生成分类实验','',
               '**当前阶段：'+phase+'**','',
               '更新时间（UTC）：'+self.state['updated'],
               '','目标：同一个无外部标签输入的ODE统一生成与反向分类，生成FID相对强类别条件基线退化暂定≤10%。',
               '', '训练使用同一35.75M参数U-Net、从头初始化、仅水平翻转、batch128、FP32。',
               '公开权重仅用于质量与指标校验，不用于初始化主对照实验。',
               '', '## 已完成评估','',
               '| 实验 | 训练步数 | 生成数 | FID / 50k训练参考 | 生成类别准确率 | 反向分类准确率 |',
               '|---|---:|---:|---:|---:|---:|']
        for name,r in self.state['results'].items():
            acc=lambda k: ('%.2f%%'%(100*r[k])) if k in r else '—'
            lines.append('| %s | %s | %s | %.4f | %s | %s |'%(name,r.get('checkpoint_step','—'),r['generated_images'],r['fid_train_50k_reference'],acc('generated_label_accuracy'),acc('reverse_accuracy')))
        lines+=['','## 运行信息','',f"- 主进程 PID：{os.getpid()}",
                '- 日志：本目录 `*.log`；训练进度：各方法目录 `progress.json`。',
                '- 协议、checkpoint SHA256、参考统计、NFE及原始样本保存在各评估目录。',
                '- `RESULTS.md` 在首轮完整比较后生成；未完成的行不会填写推测数值。',
                '- 当前seed的10%判断仅为暂定检查；完整结论还需重复seed和类别一致性。']
        if details:lines+=['','```json',json.dumps(details,ensure_ascii=False,indent=2),'```']
        (self.root/'STATUS.md').write_text('\n'.join(lines)+'\n')

    def command(self,name,module,args,progress=None):
        cmd=[sys.executable,'-u','-m',module]+list(map(str,args))
        self.status(name,command=cmd,log=str(self.root/(name+'.log')))
        with (self.root/(name+'.log')).open('a') as log:
            log.write('\n# '+now()+' '+json.dumps(cmd)+'\n');log.flush()
            child=subprocess.Popen(cmd,cwd=self.repo,env=self.env,stdout=log,stderr=subprocess.STDOUT)
            self.state['child_pid']=child.pid
            while child.poll() is None:
                detail={}
                if progress and Path(progress).exists():
                    try:detail=json.loads(Path(progress).read_text())
                    except (ValueError,OSError):pass
                self.status(name,child_pid=child.pid,progress=detail)
                time.sleep(20)
            if child.returncode!=0:
                raise RuntimeError('%s failed with exit %s; see %s'%(name,child.returncode,self.root/(name+'.log')))
        self.state.pop('child_pid',None)

    def evaluate(self,name,checkpoint,count=5000,steps=30,method='heun',cross=False,reverse=0):
        out=self.root/name
        metric=out/'metrics.json'
        if not metric.exists() or json.loads(metric.read_text())['generated_images']!=count:
            args=['--checkpoint',checkpoint,'--output',out,'--cache',self.root/'reference-cache',
                  '--samples',count,'--steps',steps,'--method',method,'--reverse-images',reverse]
            if cross:args+=['--cross-check']
            self.command(name,'strong_rf.evaluate',args,out/'progress.json')
        result=json.loads(metric.read_text())
        self.state['results'][name]=result
        self.status('评估完成：'+name)
        return result

    def train(self,kind,target):
        out=self.root/kind
        checkpoint=out/f'model-{target}.pt'
        if not checkpoint.exists():
            args=['--output',out,'--kind',kind,'--steps',target,'--resume']
            self.command('train-'+kind+'-'+str(target),'strong_rf.train',args,out/'progress.json')
        return checkpoint

    def run(self):
        official=self.root/'weights/icfm-400000.pt'
        # First small-sample check, then the full published sample count.
        pilot=self.evaluate('official-icfm-dopri5',official,count=5000,method='dopri5',cross=True)
        if pilot['fid_train_50k_reference']>15:
            raise RuntimeError('Official 5k pilot FID >15: investigate before training')
        if pilot['fid_cross_implementation_absolute_difference']>1:
            raise RuntimeError('FID implementations differ by >1: investigate reference/extractor')
        official_full=self.evaluate('official-icfm-dopri5',official,count=50000,method='dopri5',cross=True)
        if official_full['official_cleanfid_legacy_tensorflow_train']>8:
            raise RuntimeError('Official 50k FID >8: public quality reproduction not accepted')
        h30=self.evaluate('official-icfm-heun30',official,count=5000,steps=30)
        sampler_steps=30
        if h30['fid_train_50k_reference']>pilot['fid_train_50k_reference']+.5:
            sampler_steps=60
            h60=self.evaluate('official-icfm-heun60',official,count=5000,steps=60)
            if h60['fid_train_50k_reference']>pilot['fid_train_50k_reference']+1:
                sampler_steps=100
                h100=self.evaluate('official-icfm-heun100',official,count=5000,steps=100)
                if h100['fid_train_50k_reference']>pilot['fid_train_50k_reference']+1:
                    raise RuntimeError('Fixed-step solver still differs materially at 100 steps; inspect before training')
        write_json(self.root/'sampler-decision.json',{'steps':sampler_steps,'method':'heun',
                  'reason':'public same-seed 5k fixed-step comparison against dopri5; all trained methods share NFE'})
        baseline=None;budget=None
        for target in [25000,50000,100000,200000,400000]:
            checkpoint=self.train('label',target)
            probe=self.evaluate(f'label-{target}-5k',checkpoint,steps=sampler_steps)
            # A conservative project milestone, not a claim of state-of-the-art.
            if probe['fid_train_50k_reference']<=9.5 and probe.get('generated_label_accuracy',0)>=.80:
                full=self.evaluate(f'label-{target}-50k',checkpoint,count=50000,steps=sampler_steps)
                if full['fid_train_50k_reference']<=6:
                    baseline=full;budget=target;break
        if baseline is None:
            raise RuntimeError('Baseline not accepted by 400k; stop expansion and inspect architecture/training')
        write_json(self.root/'baseline-accepted.json',{'training_steps':budget,'sampler_steps':sampler_steps,
                   'metrics':baseline,'criterion':'FID50k<=6, generated target accuracy>=80%; project strong-baseline gate, not paper-equivalent quality'})
        # Same architecture, initialization seed, batches, training-image budget and sampler.
        comparison={'label':baseline}
        for kind in ['gaussian','gaussian_ot','gaussian_ot_ce','label_ot']:
            checkpoint=self.train(kind,budget)
            probe=self.evaluate(f'{kind}-{budget}-5k',checkpoint,steps=sampler_steps,
                                reverse=1000 if kind.startswith('gaussian') else 0)
            if probe['fid_train_50k_reference']>40:
                comparison[kind]=dict(probe,formal_50k_skipped='pilot FID >40; failed generation quality')
                continue
            comparison[kind]=self.evaluate(f'{kind}-{budget}-50k',checkpoint,count=50000,
                steps=sampler_steps,reverse=10000 if kind.startswith('gaussian') else 0)
        write_json(self.root/'comparison.json',comparison)
        baseline_fid=baseline['fid_train_50k_reference']
        rows=[]
        for kind,r in comparison.items():
            relative=r['fid_train_50k_reference']/baseline_fid-1
            rows.append(f"| {kind} | {r['generated_images']} | {r['fid_train_50k_reference']:.4f} | {relative:+.1%} | {r.get('reverse_accuracy','—')} |")
        text=['# 首轮强基线与统一ODE比较','',f'共同训练预算：{budget}步×128张；共同采样：Heun {sampler_steps}步。',
              '', '| 方法 | 生成数 | FID（50k训练参考） | 相对label基线 | 反向准确率 |',
              '|---|---:|---:|---:|---:|']+rows+['',
              '低频Gaussian sigma=0.5，label sigma=1；条件入口与噪声尺度共同变化，不能只归因于低频中心。',
              'Gaussian、Gaussian+OT与Gaussian+OT+CE之间控制噪声尺度，逐项比较OT与CE。',
              '这是一轮固定训练seed和生成seed的结果；10%仅是暂定判定，最终需要重复seed。',
              '未通过小样本质量检查的方法只报告5k，不将它与50k结果作正式优劣判定。',
              '不存在通过强基线质量门槛就保证统一方法成功的假设；若CE损害生成，应据此研究分类—生成权衡。']
        (self.root/'RESULTS.md').write_text('\n'.join(text)+'\n')
        self.status('首轮比较完成；查看 RESULTS.md，尚需重复seed与噪声尺度对照')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    study=Study(args.root)
    with (study.root/'study.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:study.run()
        except BaseException as error:
            study.status('运行停止，需要检查记录的原因',error=str(error),traceback=traceback.format_exc())
            raise


if __name__=='__main__':main()
