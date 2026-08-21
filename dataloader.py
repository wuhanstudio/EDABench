import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class LibrelaneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = Path(root_dir)

        if self.root_dir.name.startswith("run_") and self.root_dir.is_dir():
            self.run_dirs = [self.root_dir]
        else:
            self.run_dirs = []
            subfolders = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
            for subfolder in subfolders:
                self.run_dirs.extend(sorted(p for p in subfolder.glob("run_*") if p.is_dir()))

        if not self.run_dirs:
            raise FileNotFoundError(
                f"No run_* directories found under {self.root_dir} or its subfolders"
            )

    def __len__(self):
        return len(self.run_dirs)

    def __getitem__(self, index):
        run_dir = self.run_dirs[index]
        feature = self.feature_data(run_dir)
        label = self.label_data(run_dir)

        if self.transform:
            if random.random() > 0.5:
                feature = torch.flip(feature, dims=[2])
                label = torch.flip(label, dims=[2])
            if random.random() > 0.5:
                feature = torch.flip(feature, dims=[1])
                label = torch.flip(label, dims=[1])

        return feature, label

    def feature_data(self, run_dir):
        channels = []
        for image_name in ("pin_heatmap.png", "placement_heatmap.png", "rudy_heatmap.png"):
            image_path = run_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing required feature image: {image_path}")

            array = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
            channels.append(array)

        feature = np.stack(channels, axis=0)
        return torch.from_numpy(feature).type(torch.float32)

    def label_data(self, run_dir):
        label_path = run_dir / "routing_gt_heatmap.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing required label image: {label_path}")

        label = np.array(Image.open(label_path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(label).unsqueeze(0).type(torch.float32)


if __name__ == "__main__":
    import os
    import cv2

    # Example usage
    dataset = LibrelaneDataset("datasets/training/")
    print('len', len(dataset))

    feat, label = dataset[0]
    print('feature_shape', tuple(feat.shape), 'dtype', feat.dtype)
    print('label_shape', tuple(label.shape), 'dtype', label.dtype)
    print('feature_minmax', float(feat.min()), float(feat.max()))
    print('label_minmax', float(label.min()), float(label.max()))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: features shape {features.shape}, labels shape {labels.shape}")
        if batch_idx >= 1:  # Just show a couple of batches for demonstration
            break
