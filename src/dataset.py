import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

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

# TRAIN TRANSFORMS https://docs.pytorch.org/vision/stable/transforms.html
# https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
train_transforms = v2.Compose([ # data augmentation for train images
    #v2.Grayscale(num_output_channels=1), MobileNetV3 not friendly with grayscale
    v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    # v2.RandomVerticalFlip(p=0.2),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomRotation(degrees=(0, 45)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# VALIDATION + TEST TRANSFORMS
VT_transforms = v2.Compose([ # data normalization for validation and test images
    #v2.Grayscale(num_output_channels=1),
    v2.Resize((128,128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

#VISUALIZE OUTPUT OF TRANSFORMS
# import matplotlib.pyplot as plt
# import torchvision
# import numpy as np
# dataiter = iter(train_dl)
# images, labels = next(dataiter)
# grid = torchvision.utils.make_grid(images[:25], nrow=5, padding=2, normalize=True)
# print(f"Image shape: {images[0].shape}")  # Should be (1, 128, 128)
# grid_np = grid.numpy()
# grid_np = np.transpose(grid_np, (1, 2, 0))  # shape: (H, W, C)
# if grid_np.shape[2] == 1:
#     grid_np = grid_np[:, :, 0]  # shape: (H, W)
#     plt.imshow(grid_np, cmap='gray')
# else:
#     plt.imshow(grid_np)  # RGB fallback
# plt.axis('off')
# plt.title("Transformed Training Batch")
# plt.show()
