"""Five independent, resumable RF ablations; one sequential train/eval worker per GPU."""
import argparse
import datetime
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import write_json

KINDS = ['label', 'gaussian', 'gaussian_ot', 'gaussian_ce', 'gaussian_ot_ce']
REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def validate_resume(path, kind, start_step, target):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model', 'ema', 'optimizer', 'rng_cpu', 'rng_cuda', 'rng_numpy', 'step', 'config'}
    if not required.issubset(checkpoint):
        raise ValueError('continuation requires a full optimizer/RNG checkpoint')
    if checkpoint['config']['kind'] != kind:
        raise ValueError('continuation checkpoint kind mismatch')
    if not start_step <= checkpoint['step'] <= target:
        raise ValueError('continuation checkpoint step outside the requested interval')
    return int(checkpoint['step'])


def worker(args):
    root = Path(args.root).resolve()
    kind = args.worker
    state_path = root / kind / 'worker.json'
    state = dict(kind=kind, gpu=KINDS.index(kind), pid=os.getpid(), started=now(),
                 start_step=args.start_step, target=args.target)

    def status(phase, **details):
        state.update(phase=phase, updated=now(), **details)
        write_json(state_path, state)

    def command(phase, module, flags):
        cmd = [sys.executable, '-u', '-m', module] + list(map(str, flags))
        status(phase, command=cmd)
        with (root / kind / (phase + '.log')).open('a') as log:
            log.write('\n' + now() + ' ' + json.dumps(cmd) + '\n'); log.flush()
            child = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError('%s exited %s; inspect %s' % (phase, code, log.name))
        state.pop('child_pid', None)

    def evaluate(step, samples):
        checkpoint = root / kind / ('model-%d.pt' % step)
        output = root / 'evaluations' / ('%s-%d-%dk' % (kind, step, samples // 1000))
        metric = read(output / 'metrics.json')
        reverse = 0 if kind == 'label' else 10000
        if (metric.get('generated_images') == samples and
                metric.get('checkpoint_step') == step and
                (not reverse or metric.get('reverse_images') == reverse)):
            return
        while shutil.disk_usage(root).free < 20 * 1024**3:
            status('waiting-for-disk-space', required_free_gib=20)
            time.sleep(30)
        command('eval-%d-%dk' % (step, samples // 1000), 'strong_rf.evaluate', [
            '--checkpoint', checkpoint, '--output', output,
            '--cache', args.cache, '--samples', samples, '--batch-size', 64,
            '--steps', 60, '--method', 'heun', '--reverse-images', reverse,
            '--seed', 20260912, '--classifier', args.classifier])

    try:
        handover = read(root / 'handover.json').get(kind, {}) if not args.start_step else {}
        adopted_target = args.start_step
        if args.start_step:
            validate_resume(root / kind / 'training-latest.pt', kind, args.start_step, args.target)
            evaluate(args.start_step, args.evaluation_samples)
        if handover:
            adopted_target = handover['target']
            if not (root / kind / 'handover-complete.json').exists():
                status(handover['phase'], child_pid=handover['child_pid'], command=handover['command'],
                       adopted_existing_training=True)
                proc = Path('/proc/%d/status' % handover['child_pid'])
                while proc.exists():
                    try:
                        if any(line.startswith('State:') and 'Z' in line for line in proc.read_text().splitlines()):
                            break
                    except OSError:
                        break
                    time.sleep(5)
                if not (root / kind / ('model-%d.pt' % adopted_target)).exists():
                    raise RuntimeError('Adopted training ended without the expected milestone checkpoint')
                state.pop('child_pid', None)
                write_json(root / kind / 'handover-complete.json', dict(time=now(), target=adopted_target))
            # Finish the current milestone first, then fill the common 100k comparison.
            evaluate(adopted_target, args.evaluation_samples)
            if adopted_target > 100000:
                evaluate(100000, args.evaluation_samples)
        # Re-evaluate historical milestones with the identical full-test protocol.
        for step in [50000, 100000]:
            if not args.start_step and not handover and (root / kind / ('model-%d.pt' % step)).exists():
                evaluate(step, args.evaluation_samples)
        for step in range(50000, args.target + 1, 50000):
            if step <= adopted_target:
                continue
            checkpoint = root / kind / ('model-%d.pt' % step)
            if not checkpoint.exists():
                if args.start_step:
                    validate_resume(root / kind / 'training-latest.pt', kind, args.start_step, step)
                flags = ['--output', root / kind, '--kind', kind, '--steps', step,
                         '--resume', '--save-every', 25000, '--snapshot-every', 25000,
                         '--preview-every', 25000, '--keep-previous-checkpoint']
                if kind in {'gaussian_ce', 'gaussian_ot_ce'}:
                    flags += ['--ce-logits', 'center', '--ce-weight', .001,
                              '--ce-batch', 16, '--ce-every', 4, '--ce-steps', 4]
                command('train-%d' % step, 'strong_rf.train', flags)
            evaluate(step, args.evaluation_samples)
            if args.evaluation_samples != 50000 and step in {250000, args.target}:
                evaluate(step, 50000)
        status('complete')
    except BaseException as error:
        status('failed', error=str(error), traceback=traceback.format_exc())
        raise


def report(root, changed, target=500000, kinds=None, start_step=0):
    kinds = KINDS if kinds is None else kinds
    rows = []
    for p in sorted((root / 'evaluations').glob('*/metrics.json')):
        m = read(p)
        if m:
            rows.append(dict(m, evaluation=str(p.parent.relative_to(root))))
    lines = ['# CIFAR-10 %d 组实验（目标 %dk）' % (len(kinds), target // 1000), '', '更新时间 UTC：' + now(), '',
             '累计步数 %s → %s；每卡一组；同卡按训练→评估→续训顺序运行。' % (format(start_step, ','), format(target, ',')), '',
             '| GPU | 方法 | 阶段 | 已训练 step | 最近训练秒/step |',
             '|---:|---|---|---:|---:|']
    for kind in kinds:
        gpu = KINDS.index(kind)
        s = read(root / kind / 'worker.json')
        p = read(root / kind / 'progress.json')
        lines.append('| %d | %s | %s | %s | %s |' % (
            gpu, kind, s.get('phase', 'starting'), p.get('step', '—'),
            round(p['seconds_per_step'], 3) if 'seconds_per_step' in p else '—'))
        if s.get('error'):
            lines.append('\n错误：%s: %s\n' % (kind, s['error']))
    lines += ['', '## 已完成评估', '',
              '所有 FID 参考同一原始训练集 50k；EMA，Heun60/NFE120，固定生成 seed/batch64。',
              '5k 与 50k 生成结果分开比较；Gaussian 反向分类均测试完整 10k。', '',
              '| 方法 | 训练 step | 生成数 | FID ↓ | IS ↑ | 生成类别准确率 | 反向分类准确率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for m in sorted(rows, key=lambda m: (m['generated_images'], m['checkpoint_step'], KINDS.index(m['kind']))):
        acc = lambda key: ('%.2f%%' % (m[key] * 100)) if key in m else '—'
        lines.append('| %s | %d | %d | %.4f | %.3f | %s | %s |' % (
            m['kind'], m['checkpoint_step'], m['generated_images'],
            m['fid_train_50k_reference'], m['inception_score_mean'],
            acc('generated_label_accuracy'), acc('reverse_accuracy')))
    lines += ['', '后续评估统一FID50k，与此前Label100k的4.4815使用相同生成设置；历史FID5k保留并单独标注。',
              '完整设置见 PLAN.md；图表见 analysis/。',
              '共同 Gaussian 先验 σ=0.5、中心范数12；Label σ=1，仍存在条件方式与噪声尺度混杂。',
              '单训练 seed，不把小差异当作确定收益。']
    if any(kind in {'gaussian_ce', 'gaussian_ot_ce'} for kind in kinds):
        lines.append('CE 两组从头训练、中心投影 logits、权重0.001。')
    (root / 'STATUS.md').write_text('\n'.join(lines) + '\n')
    if changed:
        write_json(root / 'metrics.json', rows)
        from .summarize_five_gpu import summarize
        summarize(root, rows, kinds=kinds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--classifier', default='runs/reference-cifar10-classifier/cifar10-classifier-best.pt')
    parser.add_argument('--target', type=int, default=500000)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--kinds', nargs='+', choices=KINDS)
    parser.add_argument('--evaluation-samples',type=int,choices=[5000,50000],default=5000)
    parser.add_argument('--worker', choices=KINDS)
    args = parser.parse_args()
    args.classifier = str(Path(args.classifier).resolve())
    if not Path(args.classifier).is_file():
        parser.error('generation scoring classifier does not exist: ' + args.classifier)
    if args.start_step < 0 or args.start_step % 50000 or args.target % 50000 or args.target <= args.start_step:
        parser.error('start/target must be multiples of 50k with 0 <= start < target')
    kinds = KINDS if args.kinds is None else args.kinds
    if len(kinds) != len(set(kinds)):
        parser.error('kinds must be unique')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        with (root / args.worker / 'worker.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args)
        return
    with (root / 'study.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = []
        for kind in kinds:
            gpu = KINDS.index(kind)
            (root / kind).mkdir(exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            log = (root / kind / 'worker.log').open('a')
            child = subprocess.Popen([sys.executable, '-u', '-m', 'strong_rf.run_five_gpu',
                '--root', str(root), '--cache', str(Path(args.cache).resolve()),
                '--classifier', args.classifier,
                '--target', str(args.target), '--start-step', str(args.start_step),
                '--evaluation-samples',str(args.evaluation_samples),
                '--worker', kind], cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            log.close()
            children.append(child)
        write_json(root / 'processes.json', dict(supervisor=os.getpid(), workers=[p.pid for p in children], started=now()))
        last_signature = None
        while True:
            signature = [(str(p), p.stat().st_mtime_ns) for p in sorted((root / 'evaluations').glob('*/metrics.json'))]
            try:
                report(root, signature != last_signature, args.target, kinds, args.start_step)
                last_signature = signature
            except Exception:
                (root / 'report-error.log').write_text(traceback.format_exc())
            if all(p.poll() is not None for p in children):
                write_json(root / 'finished.json', dict(time=now(), exit_codes=[p.returncode for p in children]))
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
