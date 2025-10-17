import os
import pandas as pd
import numpy as np
from PIL import Image
import random

metadata_path = "../data/small_test_set/small_test_set.csv"
images_root = "../data/small_test_set"
output_dir = "../data/small_test_set/processed"

# possible gray scale

df = pd.read_csv(metadata_path)

X = []
y = df.copy().drop(columns=["filename"])
y = pd.get_dummies(y, columns=["label"], drop_first=False, dtype=float)

rot_angles = [0, 90, 180, 270]

for i, row in df.iterrows():
    label = row['label']
    filename = row['filename']
    img_path = os.path.join(images_root, label, filename) #generate filepath
    
    image = Image.open(img_path) #retrieve file
    # rotate image to benefit learning (no deterministic orientation)
    image = image.rotate(random.choice(rot_angles))

    img_array = np.array(image, dtype=np.float32) / 255.0  # normalize [0,1]
    img_flat = img_array.flatten()
    X.append(img_flat)



X = np.array(X)

np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)