import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# === CONFIG ===
metadata_path = "../data/small_test_set/small_test_set.csv"
images_root = "../data/small_test_set"
output_dir = "../data/small_test_set/processed"

# resize_shape = (256, 256)  # (width, height)
convert_to_grayscale = False  # optional: reduces to 1 channel

os.makedirs(output_dir, exist_ok=True)

# === LOAD METADATA ===
df = pd.read_csv(metadata_path)
print(f"Loaded {len(df)} metadata entries")

# === PROCESS IMAGES ===
X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    label = row['label']
    filename = row['filename']

    img_path = os.path.join(images_root, label, filename)

    if not os.path.exists(img_path):
        print(f"Missing file: {img_path}")
        continue

    try:
        img = Image.open(img_path)

        if convert_to_grayscale:
            img = img.convert("L")  # grayscale
        else:
            img = img.convert("RGB")

        # img = img.resize(resize_shape)

        img_array = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]
        img_flat = img_array.flatten()

        X.append(img_flat)
        y.append(label)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# === SAVE ARRAYS ===
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print(f"\nSaved preprocessed data to {output_dir}")
