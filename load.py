import os
import shutil
import random
DATASET_DIR = "./rawdata" 
SRC_IMAGES_DIR = os.path.join(DATASET_DIR, "images") 
SRC_MASKS_DIR = os.path.join(DATASET_DIR, "masks")
OUTPUT_TRAIN_IMG = "./data/train/images"
OUTPUT_TRAIN_MSK = "./data/train/masks"
OUTPUT_VAL_IMG = "./data/validation/images"
OUTPUT_VAL_MSK = "./data/validation/masks"
SPLIT_RATIO = 0.85
def main():
    if os.path.exists("./data"):
        shutil.rmtree("./data")
    os.makedirs(OUTPUT_TRAIN_IMG, exist_ok=True)
    os.makedirs(OUTPUT_TRAIN_MSK, exist_ok=True)
    os.makedirs(OUTPUT_VAL_IMG, exist_ok=True)
    os.makedirs(OUTPUT_VAL_MSK, exist_ok=True)
    if not os.path.exists(SRC_IMAGES_DIR) or not os.path.exists(SRC_MASKS_DIR):
        print("folders not found.")
        return
    mask_map = {}
    for f in os.listdir(SRC_MASKS_DIR):
        if f.startswith('.') or not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = f.replace("_mask_", "_").replace("mask_", "").replace("_mask", "")
        key = os.path.splitext(parts)[0]
        mask_map[key] = f
    valid_pairs = []
    for f in os.listdir(SRC_IMAGES_DIR):
        if f.startswith('.') or not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = f.replace("_sat_", "_").replace("sat_", "").replace("_sat", "")
        key = os.path.splitext(parts)[0]
        if key in mask_map:
            valid_pairs.append((f, mask_map[key]))
    if not valid_pairs:
        print("\n Invalid Pairs")
        raw_imgs = sorted([f for f in os.listdir(SRC_IMAGES_DIR) if not f.startswith('.')])
        raw_msks = sorted([f for f in os.listdir(SRC_MASKS_DIR) if not f.startswith('.')])
        valid_pairs = list(zip(raw_imgs, raw_msks))
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * SPLIT_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    print(f"Splitting into {len(train_pairs)} and {len(val_pairs)}")
    train_moved = 0
    for img_file, msk_file in train_pairs:
        shutil.move(os.path.join(SRC_IMAGES_DIR, img_file), os.path.join(OUTPUT_TRAIN_IMG, img_file))
        # Save mask under the EXACT name of the image
        shutil.move(os.path.join(SRC_MASKS_DIR, msk_file), os.path.join(OUTPUT_TRAIN_MSK, img_file))
        train_moved += 1
    val_moved = 0
    for img_file, msk_file in val_pairs:
        shutil.move(os.path.join(SRC_IMAGES_DIR, img_file), os.path.join(OUTPUT_VAL_IMG, img_file))
        shutil.move(os.path.join(SRC_MASKS_DIR, msk_file), os.path.join(OUTPUT_VAL_MSK, img_file))
        val_moved += 1

    print("\nSplit Complete")
if __name__ == "__main__":
    main()