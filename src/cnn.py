import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from dataset import GarbageDataset, train_transforms, VT_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print(f"Loaded device: {torch.cuda.get_device_name(0)}") # make sure device actually loaded correctly

#! Hyperparameters
batch_size = 32

all_images = ImageFolder(root='../data/images')
num_all_images = len(all_images) # number of images in full dataset
num_train_images = int(0.7 * num_all_images) # training data is 70% of total
num_VT_images = num_all_images - num_train_images # remaining 30% of total

train_images, val_test_images = random_split(all_images,  # splitting 70/15/15 train/val/test
                                             [num_train_images,
                                              num_VT_images])

num_test_images = int(0.5 * num_VT_images) # test and validation split 50%
num_val_images = num_VT_images - num_test_images
val_images, test_images = random_split(val_test_images, # 50/50 split of original 30% split
                                       [num_val_images,
                                        num_test_images])

train_dataset = GarbageDataset(train_images, train_transforms)
val_dataset = GarbageDataset(val_images, VT_transforms)
test_dataset = GarbageDataset(test_images, VT_transforms)


train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# VISUALIZE OUTPUT OF TRANSFORMS
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


