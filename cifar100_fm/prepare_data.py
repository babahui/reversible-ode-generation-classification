"""Download and verify the two official CIFAR-100 fine-label splits."""
import argparse
from torchvision.datasets import CIFAR100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    args = parser.parse_args()
    for train, size in [(True, 50000), (False, 10000)]:
        dataset = CIFAR100(args.data_root, train=train, download=True)
        if len(dataset) != size or len(dataset.classes) != 100:
            raise RuntimeError('unexpected dataset size or class count')
        print('train' if train else 'test', len(dataset), 'classes', len(dataset.classes))


if __name__ == '__main__':
    main()
