import os
import shutil
import random

# Set paths
SOURCE_DIR = "garbage_preprocessed"
OUTPUT_DIR = "garbage_split"
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
SEED = 42

random.seed(SEED)

# Create output folders
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Go through each category/class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIOS['train'] * total)
    val_end = train_end + int(SPLIT_RATIOS['val'] * total)

    split_data = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    # Copy files to their respective folders
    for split, files in split_data.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(split_class_dir, file)
            shutil.copy2(src, dst)

print("✅ Dataset split complete!")
