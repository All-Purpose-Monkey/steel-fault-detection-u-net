import pandas as pd
import numpy as np
import functions as fn

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("images/train.csv")  # adjust path if needed


# =========================================================
# 1. MASK COUNT PER CLASS
# =========================================================
print("\n===== MASKS PER CLASS =====")
class_counts = df["ClassId"].value_counts().sort_index()
print(class_counts)


# =========================================================
# 2. IMAGE LEVEL SUMMARY
# =========================================================
df["has_mask"] = df["EncodedPixels"].notna() & (df["EncodedPixels"] != "")

image_summary = df.groupby("ImageId").agg(
    total_masks=("ClassId", "count"),
    num_classes=("ClassId", "nunique"),
    has_any_mask=("has_mask", "max")
)

image_summary["is_empty"] = ~image_summary["has_any_mask"]

print("\n===== IMAGE SUMMARY =====")
print(image_summary["is_empty"].value_counts().rename(
    index={True: "empty_images", False: "images_with_masks"}
))

print("\nTotal images:", len(image_summary))


# simple distribution table
print("\n===== MASK DISTRIBUTION PER IMAGE =====")
print(pd.crosstab(image_summary["is_empty"], image_summary["total_masks"]))


# =========================================================
# 3. AVERAGE MASK SIZE PER CLASS
# =========================================================
mask_sizes = []

for _, row in df.iterrows():
    if pd.isna(row["EncodedPixels"]) or row["EncodedPixels"] == "":
        continue

    mask = fn.rle_decode(row["EncodedPixels"], shape=(256, 1600))
    size = mask.sum()

    mask_sizes.append((row["ImageId"], row["ClassId"], size))

size_df = pd.DataFrame(mask_sizes, columns=["ImageId", "ClassId", "size"])


print("\n===== AVERAGE MASK SIZE PER CLASS =====")
print(size_df.groupby("ClassId")["size"].mean())


print("\n===== GLOBAL MASK STATS =====")
print("Mean size:", size_df["size"].mean())
print("Median size:", size_df["size"].median())
print("Max size:", size_df["size"].max())
print("Min size:", size_df["size"].min())
