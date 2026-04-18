import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import functions as fn

# -------------------------
# DATA
# -------------------------
train_data = os.path.expanduser(
    '~/desktop/Steel-Image-Fault-Detection/images/train_images'
)
train_labels = os.path.expanduser(
    '~/desktop/Steel-Image-Fault-Detection/images/train.csv'
)

data = fn.build_training_dataset(train_labels, train_data)



# =========================================================
# AUGMENTATION
# =========================================================
def augment(img, mask):
    # brightness
    alpha = np.random.uniform(0.9, 1.1)
    img = img * alpha

    # noise
    noise = np.random.normal(0, 0.01, img.shape)
    img = img + noise

    img = np.clip(img, 0, 1)
    return img, mask


# =========================================================
# DATASET
# =========================================================
class SteelDataset(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=8, image_size=(256, 1600, 1), shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return len(self.data) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        imgs, masks = [], []

        for i in batch_idx:
            img_path, mask = self.data[i]

            img = tf.keras.utils.load_img(
                img_path,
                color_mode="grayscale",
                target_size=self.image_size[:2]
            )

            img = tf.keras.utils.img_to_array(img).astype(np.float32) / 255.0
            mask = np.array(mask).astype(np.float32)

            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)

            # augmentation only during training (shuffle=True)
            if self.shuffle:
                img, mask = augment(img, mask)

            imgs.append(img)
            masks.append(mask)

        return np.array(imgs), np.array(masks)


# =========================================================
# MODEL
# =========================================================
def build_unet(input_shape=(256, 1600, 1), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)

    def conv_block(x, f):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    b = conv_block(p4, 256)

    u1 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    u1 = tf.keras.layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 128)

    u2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c5)
    u2 = tf.keras.layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 64)

    u3 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c6)
    u3 = tf.keras.layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 32)

    u4 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c7)
    u4 = tf.keras.layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 16)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation=None)(c8)

    return tf.keras.Model(inputs, outputs)


# =========================================================
# LOSS + METRIC
# =========================================================
def dice_coef(y_true, y_pred, smooth=1):
    y_pred = tf.nn.sigmoid(y_pred)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    return bce + dice_loss(y_true, y_pred)


# =========================================================
# CALLBACKS
# =========================================================
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# =========================================================
# MODEL BUILD + COMPILE
# =========================================================
model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss=combined_loss,
    metrics=[dice_coef]
)


# =========================================================
# TRAINING PIPELINE
# =========================================================
def run_training_pipeline(data, epochs=50, batch_size=8, workers=2):

    train_size = int(0.8 * len(data))
    train_data_split = data[:train_size]
    val_data_split = data[train_size:]

    train_gen = SteelDataset(train_data_split, batch_size=batch_size, shuffle=True)
    val_gen = SteelDataset(val_data_split, batch_size=batch_size, shuffle=False)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )

    return model, history


    


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    #model, history = run_training_pipeline(data)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    #model.save("steel_unet_model.keras")
    #model.save_weights("steel_unet_weights.weights.h5")

    #print("\n✅ Model saved successfully!")

    # Save history to CSV in root directory
    #history_df = pd.DataFrame(history.history)
    #history_df.to_csv("unet4_16_training_history.csv", index=False)
    history = pd.read_csv("unet4_16_training_history.csv")
    plt.plot(history["dice_coef"])
    plt.plot(history["val_dice_coef"])
    plt.title("Dice Score")
    plt.legend(["Train", "Val"])
    plt.show()
