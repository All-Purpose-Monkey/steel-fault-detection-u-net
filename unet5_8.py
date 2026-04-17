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
#reducing augmentation intensity to prevent creating accidental defects and improve generalization - just a hunch tbh
def augment(img, mask):
    # brightness
    alpha = np.random.uniform(0.95, 1.05)
    img = img * alpha

    # noise
    noise = np.random.normal(0, 0.005, img.shape)
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


    def encoder_block(inputs, num_filters):

        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        
        return x
    
    def decoder_block(inputs, skip_features, num_filters):

        x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='valid')(inputs)

        skip_features = tf.keras.layers.Resizing(x.shape[1], x.shape[2])(skip_features)

        x = tf.keras.layers.Concatenate()([x, skip_features])

        x = tf.keras.layers.Conv2D(num_filters, 3, padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='valid')(x)
        x = tf.keras.layers.Activation('relu')(x)

        return x
    
    #encoder
    c1 = encoder_block(inputs, 8)
    c2 = encoder_block(c1, 16)
    c3 = encoder_block(c2, 32)
    c4 = encoder_block(c3, 64)
    c5 = encoder_block(c4, 128)

    #bottleneck
    b1 = tf.keras.layers.Conv2D(256, 3, padding='valid')(c4)
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(256, 3, padding='valid')(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)

    #decoder
    c6 = decoder_block(b1, c5, 128)
    c7 = decoder_block(c6, c4, 64)
    c8 = decoder_block(c7, c3, 32)
    c9 = decoder_block(c8, c2, 16)
    c10 = decoder_block(c9, c1, 8)


    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation=None)(c10)

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
    patience=5,
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
    model, history = run_training_pipeline(data)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    model.save("steel_unet_model.keras")
    model.save_weights("steel_unet_weights.weights.h5")

    print("\n✅ Model saved successfully!")

    # Save history to CSV in root directory
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    plt.plot(history.history["dice_coefficient"])
    plt.plot(history.history["val_dice_coefficient"])
    plt.title("Dice Score")
    plt.legend(["Train", "Val"])
    plt.show()
