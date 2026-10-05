"""Download and verify the official CIFAR-10 archive with torchvision."""
import argparse
from torchvision.datasets import CIFAR10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    args = parser.parse_args()
    for train in (True, False):
        data = CIFAR10(args.data_root, train=train, download=True)
        print(('train' if train else 'test'), len(data))


if __name__ == '__main__':
    main()
