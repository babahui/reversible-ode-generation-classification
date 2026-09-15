"""Export completed local evaluations as a small, auditable repository snapshot."""
import argparse
import csv
import datetime
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import sha256, write_json


KINDS = ['label', 'gaussian', 'gaussian_ot', 'gaussian_ce', 'gaussian_ot_ce']
NAMES = ['Label-conditioned', 'Low-frequency Gaussian-only', 'Gaussian + OT',
         'Gaussian + reverse CE', 'Gaussian + OT + reverse CE']


def read(path):
    return json.loads(path.read_text())


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = {}
    groups = ['cifar10-five-gpu-500k-20260913', 'cifar10-label-gaussian-1m-20260914']
    configs = destination.parents[1] / 'configs' / 'strong-rf'
    for group in groups:
        root = source / 'runs' / group
        for path in sorted((root / 'evaluations').glob('*/metrics.json')):
            metric = read(path)
            kind = metric['kind']
            if kind not in KINDS or metric['generated_images'] != 50000:
                continue
            if kind != 'label' and metric.get('reverse_images') != 10000:
                continue
            protocol = read(path.with_name('protocol.json'))
            if protocol['method'] != 'heun' or protocol['steps'] != 60:
                raise ValueError('unexpected protocol: ' + str(path))
            key = (kind, metric['checkpoint_step'])
            if key in rows:
                if rows[key]['protocol']['checkpoint_sha256'] != protocol['checkpoint_sha256']:
                    raise ValueError('same kind/step has different checkpoints')
                continue
            name = '{}-{}'.format(*key)
            output = destination / 'evaluations' / name
            output.mkdir(parents=True, exist_ok=True)
            portable = dict(protocol)
            for field in ['checkpoint', 'output', 'data_root', 'cache', 'classifier']:
                if field in portable:
                    portable[field] = Path(portable[field]).name
            portable['path_note'] = 'Original server paths reduced to basenames; numerical settings and checkpoint hash unchanged.'
            write_json(output / 'metrics.json', metric)
            write_json(output / 'protocol.json', portable)
            shutil.copyfile(path.with_name('generated-by-class.png'), output / 'generated-by-class.png')
            config = read(root / kind / 'config.json')
            config['output'] = 'runs/strong-rf/' + kind
            config['data_root'] = './data'
            config['resume'] = False
            config['steps'] = 500000
            write_json(configs / (kind + '.json'), config)
            rows[key] = dict(metric, protocol=portable, source_run=group,
                             evaluation='evaluations/' + name)
    ordered = [rows[key] for key in sorted(rows, key=lambda k: (k[1], KINDS.index(k[0])))]
    write_json(destination / 'metrics.json', ordered)
    columns = ['kind', 'step', 'generated_images', 'generated_label_accuracy', 'reverse_accuracy',
               'fid_train_50k_reference', 'inception_score_mean', 'inception_score_std',
               'kid_legacy_2560_vs_test5k_mean', 'kid_legacy_2560_vs_test5k_std', 'checkpoint_sha256']
    with (destination / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in ordered:
            record = {k: row.get(k, '') for k in columns}
            kid = row.get('kid_legacy_2560_vs_test5k', {})
            record.update(step=row['checkpoint_step'], checkpoint_sha256=row['protocol']['checkpoint_sha256'],
                          kid_legacy_2560_vs_test5k_mean=kid.get('kernel_inception_distance_mean', ''),
                          kid_legacy_2560_vs_test5k_std=kid.get('kernel_inception_distance_std', ''))
            writer.writerow(record)
    common = set.intersection(*[{step for kind, step in rows if kind == k} for k in KINDS])
    latest_common = max(common)
    latest = [max((r for r in ordered if r['kind'] == kind), key=lambda r: r['checkpoint_step']) for kind in KINDS]
    classifier = source / 'runs/reference-cifar10-classifier/cifar10-classifier-best.pt'
    shutil.copyfile(classifier, destination / 'reference-classifier.pt')
    write_json(destination / 'reference-classifier.json', dict(
        sha256=sha256(classifier), architecture='CIFAR10ReferenceClassifier, small_resnet',
        epoch=44, test_accuracy=0.9236, role='generated image scoring only; never ODE guidance'))
    source_files = ['strong_rf/train.py', 'strong_rf/common.py', 'strong_rf/evaluate.py',
                    'unified_transport/latent.py', 'unified_transport/ot_pairing.py',
                    'unified_transport/reference_classifier.py']
    write_json(destination / 'source-provenance.json', dict(
        exported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_runs=groups, original_source_sha256={p: sha256(source / p) for p in source_files},
        note='Source hashes before publication-only CLI/documentation additions. Metrics are not recomputed.'))

    def table(selected):
        lines = ['| Method | Step | Generated accuracy | Reverse accuracy | FID | IS |',
                 '|---|---:|---:|---:|---:|---:|']
        for row in selected:
            rev = '{:.2f}%'.format(100*row['reverse_accuracy']) if 'reverse_accuracy' in row else 'N/A'
            lines.append('| {} | {}K | {:.2f}% | {} | {:.4f} | {:.3f} |'.format(
                NAMES[KINDS.index(row['kind'])], row['checkpoint_step']//1000,
                100*row['generated_label_accuracy'], rev,
                row['fid_train_50k_reference'], row['inception_score_mean']))
        return lines

    lines = ['# CIFAR-10: New-backbone Results', '',
             'Frozen export of completed evaluations, not live job status. No 1M result is available in this snapshot.', '',
             '## Matched training budget: {}K'.format(latest_common//1000), '']
    lines += table([rows[(k, latest_common)] for k in KINDS])
    lines += ['', '## Latest completed checkpoint per method', '',
              '**Different step counts: these rows are not a matched-budget ranking.**', '']
    lines += table(latest)
    lines += ['', '## Protocol and interpretation', '',
              '- EMA, Heun 60 steps / NFE120, 50,000 generated images, batch64, generation seed20260912.',
              '- FID references 50,000 original CIFAR-10 training images with torch-fidelity FP32 Inception features.',
              '- Reverse accuracy uses all 10,000 real test images, with no label or external classifier passed to the ODE.',
              '- Generation accuracy uses the included fixed evaluation classifier, not ground truth.',
              '- IS mean is shown; its split standard deviation and all confusion matrices are in the raw JSON.',
              '- KID in CSV/JSON is the legacy **2560 generated / 5000 test reference** protocol, not KID50K.',
              '- Single training seed; Label sigma=1 versus Gaussian sigma=0.5. No SOTA or 95% reverse-accuracy claim.',
              '- Reported CE uses center-projection logits, weight0.001, every4 updates, batch16, Euler4.',
              '- No posterior head or history adapter is used. The new history comparison remains planned.',
              '- These are a different backbone and protocol from the older repository results; do not merge their FID columns.', '',
              '[Full metrics CSV](metrics.csv) | [Full metrics JSON](metrics.json) | [Implementation and commands](../../strong_rf/README.md)', '',
              '## Generated-image comparisons', '',
              'Each panel uses the first ten samples per class from the formal evaluation, without quality selection.',
              'Rows follow CIFAR-10 IDs: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.',
              'Columns within a panel are sample indices. Seeds/batches match across runs; Label and Gaussian initial distributions differ.', '']
    for step in sorted({200000, latest_common, 550000}):
        selected = [rows[(k, step)] for k in KINDS if (k, step) in rows]
        if not selected:
            continue
        width = 370
        canvas = Image.new('RGB', (width*len(selected), 450), 'white')
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for i, row in enumerate(selected):
            x = i*width
            draw.text((x+8, 8), row['kind'], fill='black', font=font)
            draw.text((x+8, 25), '{}K; gen {:.2f}%; FID {:.2f}'.format(step//1000,
                      100*row['generated_label_accuracy'], row['fid_train_50k_reference']), fill='black', font=font)
            with Image.open(destination / row['evaluation'] / 'generated-by-class.png') as img:
                canvas.paste(img.convert('RGB'), (x+8, 55))
        filename = 'comparison-{}k.png'.format(step//1000)
        canvas.save(destination / filename)
        lines += ['### {}K ({} methods)'.format(step//1000, len(selected)), '',
                  '![{}K generated grids]({})'.format(step//1000, filename), '']
    lines += ['## Checkpoints', '',
              'The flow weights and full optimizer states are not committed (each EMA is about137MiB).',
              'Retrain with the reproduction commands, or request the weights matching the SHA256 in each protocol JSON.',
              'The exact evaluation classifier is included as `reference-classifier.pt`; its fingerprint is in `reference-classifier.json`.',
              'Never load untrusted pickle checkpoints.', '']
    (destination / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', required=True)
    parser.add_argument('--output', default='results/strong-rf')
    args = parser.parse_args()
    export(Path(args.source_repo).resolve(), Path(args.output).resolve())


if __name__ == '__main__':
    main()
