import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

# === CONFIG ===
metadata_path = "../data/metadata.csv"            # path to your metadata CSV
source_root = "../data/images"                  # folder containing class subfolders
subset_root = "../data/small_test_set"         # destination folder for subset
subset_metadata_path = os.path.join(subset_root, "small_test_set.csv")
subset_fraction = 0.1                     # 10% of each class
random_state = 42

# === LOAD METADATA ===
df = pd.read_csv(metadata_path)

# Ensure your CSV has 'filename' and 'label' columns
if not {'filename', 'label'}.issubset(df.columns):
    raise ValueError("CSV must have columns named 'filename' and 'label'.")

print(f"Loaded {len(df)} entries from metadata")

# === STRATIFIED SAMPLING ===
subset_df, _ = train_test_split(
    df,
    stratify=df['label'],
    test_size=1 - subset_fraction,
    random_state=random_state
)

print(f"Selected {len(subset_df)} images ({subset_fraction*100:.0f}% per label)")

# === COPY FILES ===
os.makedirs(subset_root, exist_ok=True)

for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Copying images"):
    cls = row['label']
    filename = row['filename']

    # Build paths
    src = os.path.join(source_root, cls, filename)
    dst_dir = os.path.join(subset_root, cls)
    dst = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"Missing file: {src}")

# === SAVE SUBSET METADATA ===
subset_df.to_csv(subset_metadata_path, index=False)
print(f"\nSubset saved to: {subset_root}")
print(f"Metadata saved to: {subset_metadata_path}")
