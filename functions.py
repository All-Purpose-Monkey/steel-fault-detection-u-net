import os
import sklearn as sl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

# =========================
# RLE → MASK
# =========================
def rle_decode(rle_string, shape):
    """
    Convert RLE string to a binary mask.

    Args:
        rle_string (str): run-length as string formatted (start length)
        shape (tuple): (height, width)

    Returns:
        np.array: 2D mask
    """
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]

    starts = np.array(starts) - 1  # convert to zero-index
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    # reshape (IMPORTANT: column-wise order)
    return img.reshape(shape, order='F')


# =========================
# MASK → RLE
# =========================
def rle_encode(mask):
    """
    Convert binary mask to RLE string.

    Args:
        mask (np.array): 2D binary mask

    Returns:
        str: run-length encoding
    """
    pixels = mask.flatten(order='F')

    # Add padding to catch transitions
    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def build_training_dataset(csv_path, image_dir, shape=(256, 1600)):
    """
    Returns:
        list of (image_path, mask, class_id)
    """
    df = pd.read_csv(csv_path)

    dataset = {}

    for _, row in df.iterrows():
        image_id = row['ImageId']
        class_id = int(row['ClassId'])
        rle = row['EncodedPixels']

        img_path = os.path.join(image_dir, image_id)

        if img_path not in dataset:
            dataset[img_path] = {
                "mask": np.zeros(shape, dtype=np.uint8),
                "class_ids": set()
            }

        class_mask = rle_decode(rle, shape)

        # binary union of all defect regions
        dataset[img_path]["mask"][class_mask == 1] = 1

        # store all class_ids seen for this image
        dataset[img_path]["class_ids"].add(class_id)

    return [
        (img_path, v["mask"], sorted(list(v["class_ids"])))
        for img_path, v in dataset.items()
    ]
def show_sample(dataset, idx=0):
    img_path, mask, class_id = dataset[idx]

    image = Image.open(img_path).convert("RGB")

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title(f"Image (class {class_id})")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()
