"""Sequential generation/metrics evaluation; can resume at CFG boundaries."""
import argparse
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image
from .config import LATENT_SHAPE
from .data import ImageNetImages
from .flow import indexed_noise, integrate
from .model import build_model
from .rae import Decoder
from .state import atomic_save, read_json, write_json, now


class EvaluationInterrupted(Exception):
    pass


def check_stop(stop):
    if stop():
        raise EvaluationInterrupted('Evaluation interrupted; training checkpoint is already saved')


class Features:
    def __init__(self, device, inception_weights=None, classifier_weights=None):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        kwargs = {'feature_extractor_weights_path': inception_weights} if inception_weights else {}
        self.inc = FeatureExtractorInceptionV3('inception-v3-compat', ['2048'], **kwargs).to(device).eval()
        if classifier_weights:
            self.clf = resnet50(weights=None)
            self.clf.load_state_dict(torch.load(classifier_weights, map_location='cpu', weights_only=True))
        else:
            self.clf = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.clf.to(device).eval().requires_grad_(False)
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    @torch.no_grad()
    def __call__(self, images):
        pixels = (images.clamp(0, 1) * 255).round().to(torch.uint8)
        return self.inc(pixels)[0].cpu(), self.clf(self.transform(pixels)).float().argmax(1).cpu()


@torch.no_grad()
def evaluate(model, config, args, step, classes, stop=lambda: False):
    started = time.monotonic()
    ec = config['evaluation']
    device = next(model.parameters()).device
    out = Path(args.output) / 'evaluations' / f'step-{step:09d}'
    protocol = dict(step=step, evaluation=ec, classes=classes,
                    reference_root=str(Path(args.data_root).resolve()),
                    metric='inception-v3-compat FP32, TF32 disabled',
                    preprocessing='resize-short-side-256, center-crop-256',
                    noise='CPU per-sample generator seed=seed+index*1000003',
                    subset_check_only=args.allow_subset)
    old = read_json(out / 'protocol.json')
    if old is not None and old != protocol:
        raise ValueError('Evaluation protocol changed; use a separate output directory')
    if (out / 'complete.json').exists():
        return
    write_json(out / 'protocol.json', protocol)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    features = Features(device, args.inception_weights, args.classifier_weights)
    val = ImageNetImages(args.data_root, 'val', classes=classes, allow_subset=args.allow_subset)
    ref_path = Path(args.output) / 'references/imagenet-val.pt'
    signature = dict(dataset=val.signature(), preprocessing=protocol['preprocessing'], metric=protocol['metric'])
    if ref_path.exists():
        reference = torch.load(ref_path, map_location='cpu', weights_only=False)
        if reference['signature'] != signature:
            raise ValueError('Validation reference cache does not match this dataset/protocol')
    else:
        fs, ps, ys = [], [], []
        for images, y in DataLoader(val, batch_size=32, num_workers=config['train']['num_workers'], pin_memory=True):
            check_stop(stop)
            f, p = features(images.to(device, non_blocking=True))
            fs.append(f); ps.append(p); ys.append(y)
        reference = dict(features=torch.cat(fs), predictions=torch.cat(ps), labels=torch.cat(ys), signature=signature)
        atomic_save(reference, ref_path)
    decoder = Decoder(args.rae_repo, args.weights, device)
    for scale in ec['cfg_scales']:
        target = out / f'cfg-{scale:g}'
        if (target / 'metrics.json').exists():
            continue
        fs, ps, ys, preview = [], [], [], []
        tic = time.monotonic()
        for start in range(0, ec['samples'], ec['batch_size']):
            check_stop(stop)
            indices = list(range(start, min(start + ec['batch_size'], ec['samples'])))
            labels = torch.tensor(indices, device=device) % config['model']['num_classes']
            noise = indexed_noise(indices, LATENT_SHAPE, ec['seed'], device)
            z = integrate(model, noise, labels, config['model']['num_classes'], ec['steps'], scale)
            images = torch.cat([decoder(chunk) for chunk in z.split(ec['decode_batch'])])
            f, p = features(images)
            fs.append(f); ps.append(p); ys.append(labels.cpu())
            if start < 80:
                preview.append(images[:min(80-start, len(images))].cpu())
            if start % 100 == 0:
                write_json(Path(args.output) / 'status.json', dict(phase='evaluating', step=step, cfg=scale,
                           completed=start+len(indices), total=ec['samples'], updated_utc=now()))
        target.mkdir(parents=True, exist_ok=True)
        save_image(torch.cat(preview), target / 'samples.png', nrow=10)
        generated, predicted, labels = torch.cat(fs), torch.cat(ps), torch.cat(ys)
        from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
        from torch_fidelity.metric_kid import kid_features_to_metric
        score = fid_statistics_to_metric(fid_features_to_statistics(generated),
                    fid_features_to_statistics(reference['features']), verbose=False)
        score.update(kid_features_to_metric(generated, reference['features'], kid_subsets=ec['kid_subsets'],
                     kid_subset_size=min(1000, len(generated), len(reference['features'])), rng_seed=ec['seed']+1, verbose=False))
        score.update(step=step, cfg_scale=scale, count=len(labels), reference_count=len(reference['features']),
                     generated_label_accuracy=float((predicted == labels).float().mean()),
                     real_reference_classifier_accuracy=float((reference['predictions'] == reference['labels']).float().mean()),
                     ode_steps=ec['steps'], network_evaluations_per_image=ec['steps']*(1 if scale==1 else 2),
                     elapsed_seconds=time.monotonic()-tic, subset_check_only=args.allow_subset)
        atomic_save(dict(features=generated, predictions=predicted, labels=labels), target / 'features.pt')
        write_json(target / 'metrics.json', score)
    write_json(out / 'complete.json', dict(step=step, completed_utc=now(), elapsed_seconds=time.monotonic()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--rae-repo', default='third_party/RAEv2')
    p.add_argument('--samples', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--inception-weights')
    p.add_argument('--classifier-weights')
    p.add_argument('--allow-subset', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    if args.samples is not None: config['evaluation']['samples'] = args.samples
    if args.batch_size is not None: config['evaluation']['batch_size'] = args.batch_size
    from .config import validate_config
    validate_config(config)
    model = build_model(config['model'], args.rae_repo, checkpointing=False)
    model.load_state_dict(saved['ema'], strict=True)
    step, classes = saved['step'], saved['dataset']['classes']
    del saved
    evaluate(model.cuda().eval(), config, args, step, classes)


if __name__ == '__main__':
    main()
