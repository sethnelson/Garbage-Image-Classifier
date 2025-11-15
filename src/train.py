import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from dataset import GarbageDataset, train_transforms, VT_transforms
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
import torch.optim as optim
from cnn import train_model

torch.manual_seed(45)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print(f"Loaded device: {torch.cuda.get_device_name(0)}") # make sure device actually loaded correctly

#! Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

all_images = ImageFolder(root='../data/images') # Thanks Michael V. for the tip https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
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

mobilenet_v3_large = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=6)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=learning_rate)

mlp_model, train_losses, val_losses, val_accs = train_model(mobilenet_v3_large, train_dl,
                                                            val_dl, criterion,
                                                            optimizer, epochs, device,
                                                            display=True)

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('MLP Model Training and Validation Losses')
plt.show()

print("Training finished.")
