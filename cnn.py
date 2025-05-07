import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# اندازه تصویر
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0  # برای sigmoid مناسب است

    return image, mask

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask

def create_dataset(image_dir, mask_dir, batch_size=BATCH_SIZE):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet_plus(img_size=(256, 256, 3)):
    inputs = layers.Input(shape=img_size)

    x1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(x1)

    x2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(x2)

    x3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(x3)

    x4 = conv_block(p3, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x4)
    u3 = layers.concatenate([u3, x3])
    x5 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x5)
    u2 = layers.concatenate([u2, x2])
    x6 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x6)
    u1 = layers.concatenate([u1, x1])
    x7 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x7)

    model = models.Model(inputs, outputs)
    return model

# مسیر پوشه تصاویر و ماسک‌ها را وارد کن
image_dir = os.path.expanduser("~/Documents/AI/anaconda3/envs/ML/data/img1/train_img/")
mask_dir = os.path.expanduser("~/Documents/AI/anaconda3/envs/ML/data/img1/train_lab/")

# ایجاد دیتاست
train_ds = create_dataset(image_dir, mask_dir)

# ساخت مدل
model = build_unet_plus()

# کامپایل
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# آموزش
model.fit(train_ds, epochs=2)
model.save("unet_plus.h5")

