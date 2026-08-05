"""TFRecord dataset loader for Petals to the Metal competition."""

import io
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tfrecord.torch.dataset import TFRecordDataset


def get_available_sizes(data_dir):
    """Return sorted list of available image resolutions under *data_dir*."""
    data_path = Path(data_dir)
    sizes = []
    for d in data_path.glob('tfrecords-jpeg-*x*'):
        size_str = d.name.split('-')[-1]  # e.g. '192x192'
        w, h = size_str.split('x')
        if w == h:
            sizes.append(int(w))
    return sorted(sizes)


class PetalsDataset(Dataset):
    """PyTorch Dataset for TFRecord flower images.

    Supports 192 / 224 / 331 / 512 px and auto-detects train / val / test.

    Args:
        data_dir: root data directory (contains tfrecords-jpeg-{N}x{N}/)
        image_size: image resolution (192, 224, 331, or 512)
        split: 'train', 'val', or 'test'
        transform: torchvision transform pipeline
        predecode: decode JPEG to PIL in __init__ (default False —
                   storing raw bytes is faster due to better cache locality)

    Returns (``ds[idx]``):
        +-----------+-----------------------------------+
        | split     | returns                           |
        +===========+===================================+
        | train/val | ``(Tensor, int)``                 |
        +-----------+-----------------------------------+
        | test      | ``(Tensor, str)`` — str = image_id|
        +-----------+-----------------------------------+
    """

    def __init__(self, data_dir, image_size=192, split='train', transform=None,
                 predecode=False):
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'"
            )

        self.split = split
        self.transform = transform
        self.predecode = predecode
        self.samples = []

        tfrecord_dir = (
            Path(data_dir)
            / f'tfrecords-jpeg-{image_size}x{image_size}'
            / split
        )
        tfrecord_paths = sorted(tfrecord_dir.glob('*.tfrec'))

        if not tfrecord_paths:
            raise FileNotFoundError(
                f"No .tfrec files found in '{tfrecord_dir}'"
            )

        if split == 'test':
            description = {'image': 'byte', 'id': 'byte'}
        else:
            description = {'image': 'byte', 'class': 'int'}

        total_files = len(tfrecord_paths)
        for fi, path in enumerate(tfrecord_paths, 1):
            dataset = TFRecordDataset(
                str(path), index_path=None, description=description
            )
            for record in dataset:
                if predecode and split != 'test':
                    record['image'] = Image.open(
                        io.BytesIO(record['image'])
                    ).convert('RGB')
                self.samples.append(record)
            print(f'\r  Loading {split}: file {fi}/{total_files}, '
                  f'{len(self.samples)} samples', end='', flush=True)
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]

        if self.predecode:
            img = record['image']
        else:
            img = Image.open(io.BytesIO(record['image'])).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.split == 'test':
            image_id = (record['id'].decode('utf-8')
                        if isinstance(record['id'], bytes) else record['id'])
            return img, image_id

        label = int(record['class'])
        return img, label
