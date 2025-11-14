import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.transforms import v2
from torchvision.io import decode_image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# DEFINE DATASET
class GarbageDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, index):
        image, label = self.images[index]
        return self.transforms(image), label

    def __len__(self):
        return len(self.images)

# TRAIN TRANSFORMS
train_transforms = v2.Compose([ # data augmentation for train images
    v2.Grayscale(num_output_channels=1),
    v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomPerspective(distortion_scale=0.5, p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5]),
])

# VALIDATION + TEST TRANSFORMS
VT_transforms = v2.Compose([ # data normalization for validation and test images
    v2.Grayscale(num_output_channels=1),
    v2.Resize((128,128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])
