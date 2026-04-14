import os
import sklearn as sl
import numpy as np
import pandas as pd

train_data = os.path.expanduser('~/images/train')
train_labels = os.path.expanduser('~/images/train.csv')

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


# =========================
# LOAD LABELS
# =========================
def load_labels(csv_path):
    """
    Load and parse training labels.

    Returns:
        DataFrame with columns:
        - ImageId
        - ClassId
        - EncodedPixels
    """
    df = pd.read_csv(csv_path)

    # Split "ImageId_ClassId"
    df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.split('_', expand=True)
    df['ClassId'] = df['ClassId'].astype(int)

    return df

if __name__ == "__main__":
    # fake mask
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    print("Original mask:\n", mask)
    print("RLE:", rle)
    print("Decoded mask:\n", decoded)

    assert np.array_equal(mask, decoded), "RLE encode/decode failed!"


