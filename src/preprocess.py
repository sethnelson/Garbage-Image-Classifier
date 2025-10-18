# pair programming between Seth Nelson and Braeden Watkins

import os
import pandas as pd
import numpy as np
from PIL import Image
import random

metadata_path = "../data/metadata.csv"
images_root = "../data/images"
output_dir = "../data/processed"

# possible gray scale

df = pd.read_csv(metadata_path)

X = []
y = df.copy().drop(columns=["filename"])
# OHE labels
y = pd.get_dummies(y, columns=["label"], drop_first=False, dtype=float)

# used to randomly rotate the current image before flattening
rot_angles = [0, 90, 180, 270]

# iterate through every image in the data set
for i, row in df.iterrows():
    label = row['label']
    filename = row['filename']
    img_path = os.path.join(images_root, label, filename) #generate filepath
    
    image = Image.open(img_path) #retrieve file
    # rotate image to benefit learning (try to make the model generalize better)
    image = image.rotate(random.choice(rot_angles))

    img_array = np.array(image, dtype=np.float32) / 255.0  # normalize [0,1]
    img_flat = img_array.flatten()
    X.append(img_flat)

X = np.array(X)

# save the files to a folder for use by split.py
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)