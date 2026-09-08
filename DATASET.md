# CIFAR-10 dataset

CIFAR-10 is not stored in this Git repository. The local archive is about
163 MB and the extracted Python files are about 178 MB; keeping both out of
Git makes clones small and avoids redistributing third-party data.

## Automatic download

Install the base environment first, then run:

```bash
python scripts/prepare_cifar10.py --data-root ../all_datasets
```

The script uses `torchvision.datasets.CIFAR10` and its configured CIFAR-10
source. Torchvision validates the downloaded archive before extraction. It
then verifies that both splits can be loaded and contain the expected number
of examples and classes.

Expected layout:

```text
../all_datasets/
├── cifar-10-python.tar.gz       # optional after successful extraction
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    └── test_batch
```

The archive metadata used by the installed torchvision dataset loader is:

```text
URL: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
MD5: c58f30108f718f92721af3b95e74349a
```

## Offline or shared storage

On a machine without internet access, transfer either the archive or the
extracted `cifar-10-batches-py` directory into the data root. Verify it without
attempting a download:

```bash
python scripts/prepare_cifar10.py \
  --data-root /path/to/all_datasets \
  --verify-only
```

Pass the same directory to all commands:

```bash
DATA_ROOT=/path/to/all_datasets ./scripts/train_1m.sh
DATA_ROOT=/path/to/all_datasets ./scripts/evaluate_1m.sh
```

Both release scripts run the dataset preparation check before starting. If the
data is absent they download it automatically; set `CIFAR10_OFFLINE=1` to
require existing local data and prohibit a download attempt.

## Data handling

The loader normalizes images from `[0, 1]` to `[-1, 1]`. Training enables
horizontal flips and a 32x32 random crop with four-pixel padding. Evaluation
uses deterministic resizing and normalization. Labels remain the standard
CIFAR-10 integer labels `0..9` for this ten-class experiment.
