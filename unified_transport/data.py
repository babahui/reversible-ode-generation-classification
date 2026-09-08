import random
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, STL10, ImageFolder

from .latent import OrthogonalGaussianLatent


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class _TinyImageNetSplit(Dataset):
    """Minimal Tiny-ImageNet reader with stable wnid-to-index mapping."""

    def __init__(self, root: str, train: bool, transform):
        self.root = Path(root)
        self.transform = transform
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet metadata not found: {wnids_path}")
        self.wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples = []
        if train:
            train_root = self.root / "train"
            for wnid in self.wnids:
                image_dir = train_root / wnid / "images"
                for path in sorted(image_dir.glob("*")):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | {".jpeg"}:
                        self.samples.append((path, self.class_to_idx[wnid]))
        else:
            annotation_path = self.root / "val" / "val_annotations.txt"
            image_root = self.root / "val" / "images"
            if not annotation_path.exists():
                raise FileNotFoundError(f"Tiny-ImageNet validation annotations not found: {annotation_path}")
            for line in annotation_path.read_text().splitlines():
                fields = line.split("\t")
                if len(fields) >= 2 and fields[1] in self.class_to_idx:
                    path = image_root / fields[0]
                    if path.exists():
                        self.samples.append((path, self.class_to_idx[fields[1]]))
        if not self.samples:
            split = "train" if train else "val"
            raise FileNotFoundError(f"No Tiny-ImageNet images found for {split} under {self.root}")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, target


def image_transform(
    image_size: int,
    train: bool,
    channels: int = 3,
    horizontal_flip: bool = False,
    random_crop: bool = False,
) -> transforms.Compose:
    operations = [transforms.Resize((image_size, image_size))]
    if train and random_crop:
        operations.append(transforms.RandomCrop(image_size, padding=4))
    if train and horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels),
        ]
    )
    return transforms.Compose(operations)


class FilteredVisionDataset(Dataset):
    """A consistently normalized and relabeled vision subset.

    Tiny-ImageNet is stored as ``train/<wnid>/*.JPEG`` and a flat validation
    directory with labels in ``val/val_annotations.txt``.  This class keeps
    the same local-label contract as MNIST/CIFAR: ``classes`` contains the
    original integer class indices and returned labels are 0..len(classes)-1.
    """

    def __init__(
        self,
        mode: str,
        root: str,
        train: bool,
        image_size: int,
        classes: Sequence[int],
        download: bool = False,
        random_crop: bool = False,
    ):
        if mode not in {"mnist", "cifar10", "tinyimagenet", "stl10"}:
            raise ValueError("unsupported vision dataset mode")
        max_class = 199 if mode == "tinyimagenet" else 9
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("classes must be a non-empty sequence without duplicates")
        if any(label < 0 or label > max_class for label in classes):
            raise ValueError(
                "class values must lie in [0, 9] for MNIST/CIFAR-10 or [0, 199] for Tiny-ImageNet"
            )

        self.mode = mode
        self.classes = tuple(int(label) for label in classes)
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        channels = 1 if mode == "mnist" else 3
        transform = image_transform(
            image_size,
            train=train,
            channels=channels,
            horizontal_flip=mode in {"cifar10", "tinyimagenet", "stl10"},
            random_crop=random_crop and mode in {"cifar10", "tinyimagenet", "stl10"},
        )
        if mode == "tinyimagenet":
            self.dataset = _TinyImageNetSplit(root, train=train, transform=transform)
        elif mode == "stl10":
            self.dataset = STL10(
                root=root,
                split="train" if train else "test",
                transform=transform,
                download=download,
            )
            self.dataset.targets = self.dataset.labels.tolist()
        else:
            dataset_class = MNIST if mode == "mnist" else CIFAR10
            self.dataset = dataset_class(
                root=root, train=train, transform=transform, download=download
            )
        targets = torch.as_tensor(self.dataset.targets)
        keep = torch.zeros_like(targets, dtype=torch.bool)
        for label in self.classes:
            keep |= targets == label
        self.indices = keep.nonzero(as_tuple=False).flatten().tolist()
        self.class_counts = torch.tensor(
            [int((targets == label).sum()) for label in self.classes], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_label = self.dataset[self.indices[index]]
        return image, self.class_to_index[int(original_label)]


class VisionLatentDataset(Dataset):
    """Couple labeled images to matching class-conditional latent noise."""

    def __init__(self, dataset: FilteredVisionDataset, latent: OrthogonalGaussianLatent):
        if len(dataset.classes) != latent.num_classes:
            raise ValueError("dataset class count and latent component count must match")
        self.dataset = dataset
        self.latent = latent

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        latent = self.latent.sample(torch.tensor([label])).squeeze(0)
        return torch.stack((image, latent), dim=0), label


class CIFARLatentDataset(VisionLatentDataset):
    """Backward-compatible wrapper for all ten CIFAR-10 classes."""

    def __init__(
        self,
        root: str,
        latent: OrthogonalGaussianLatent,
        train: bool,
        download: bool = False,
    ):
        dataset = FilteredVisionDataset(
            "cifar10",
            root,
            train,
            latent.image_shape[-1],
            tuple(range(latent.num_classes)),
            download,
        )
        super().__init__(dataset, latent)


class FolderMarginalsDataset(Dataset):
    """Draw an independent tuple from domain folders without a Cartesian product.

    Layout: root/domain_a/*, root/domain_b/*, ... . The epoch length is the
    largest domain size; other domain indices are independently resampled.
    """

    def __init__(
        self,
        root: str,
        domains: Sequence[str],
        image_size: int,
        train: bool = True,
    ):
        if len(domains) < 2:
            raise ValueError("at least two domain folders are required")
        self.root = Path(root)
        self.domains = tuple(domains)
        self.transform = image_transform(
            image_size, train=train, channels=3, horizontal_flip=train
        )
        self.paths: List[List[Path]] = []
        for domain in self.domains:
            folder = self.root / domain
            paths = sorted(
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise FileNotFoundError(f"no images found in {folder}")
            self.paths.append(paths)
        self.length = max(map(len, self.paths))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        images = []
        anchor_domain = index % len(self.paths)
        for domain_index, paths in enumerate(self.paths):
            if domain_index == anchor_domain:
                path = paths[(index // len(self.paths)) % len(paths)]
            else:
                path = paths[random.randrange(len(paths))]
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images, dim=0)


def load_image(
    path: str, image_size: int, channels: int, device: torch.device
) -> torch.Tensor:
    with Image.open(path) as image:
        mode = "L" if channels == 1 else "RGB"
        tensor = image_transform(
            image_size, train=False, channels=channels
        )(image.convert(mode))
    return tensor.unsqueeze(0).to(device)
