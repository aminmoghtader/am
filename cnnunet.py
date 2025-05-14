import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

# تنظیمات پیشرفته
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 100
INIT_FILTERS = 32

# 1. لود و پیش‌پردازش داده‌ها با آگمنتاسیون پیشرفته
def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return image, mask

def advanced_augment(image, mask):
    # Random flips
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Random rotation
    angle = tf.random.uniform([], -0.2, 0.2)  # ±11.5 degrees
    image = tfa.image.rotate(image, angle)
    mask = tfa.image.rotate(mask, angle, interpolation='nearest')
    
    # Color adjustments
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def create_dataset(image_dir, mask_dir, augment=True):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(advanced_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(200)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# 2. معماری پیشرفته UNet با توجه و بلوک‌های باقیمانده
class AttentionGate(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv_g = layers.Conv2D(filters, 1, padding='same')
        self.conv_x = layers.Conv2D(filters, 1, padding='same')
        self.conv_psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
    
    def call(self, g, x):
        g1 = self.conv_g(g)
        x1 = self.conv_x(x)
        psi = layers.Activation('relu')(g1 + x1)
        psi = self.conv_psi(psi)
        return x * psi

def residual_block(x, filters, use_attention=True):
    shortcut = x
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if use_attention:
        x = AttentionGate(filters)(shortcut, x)
    
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_advanced_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    e1 = residual_block(inputs, INIT_FILTERS)
    p1 = layers.MaxPooling2D()(e1)
    
    e2 = residual_block(p1, INIT_FILTERS*2)
    p2 = layers.MaxPooling2D()(e2)
    
    e3 = residual_block(p2, INIT_FILTERS*4)
    p3 = layers.MaxPooling2D()(e3)
    
    # Bridge
    b1 = residual_block(p3, INIT_FILTERS*8)
    
    # Decoder with attention
    u1 = layers.UpSampling2D()(b1)
    a1 = AttentionGate(INIT_FILTERS*4)(e3, u1)
    d1 = layers.Concatenate()([u1, a1])
    d1 = residual_block(d1, INIT_FILTERS*4)
    
    u2 = layers.UpSampling2D()(d1)
    a2 = AttentionGate(INIT_FILTERS*2)(e2, u2)
    d2 = layers.Concatenate()([u2, a2])
    d2 = residual_block(d2, INIT_FILTERS*2)
    
    u3 = layers.UpSampling2D()(d2)
    a3 = AttentionGate(INIT_FILTERS)(e1, u3)
    d3 = layers.Concatenate()([u3, a3])
    d3 = residual_block(d3, INIT_FILTERS)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d3)
    
    return models.Model(inputs, outputs)

# 3. توابع loss و معیارهای پیشرفته
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# 4. آموزش مدل
def train_model():
    # ایجاد دیتاست
    train_ds = create_dataset(image_dir, mask_dir)
    
    # ساخت مدل
    model = build_advanced_unet()
    
    # کامپایل با تنظیمات پیشرفته
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss=bce_dice_loss,
        metrics=['accuracy', iou_metric]
    )
    
    # کالبک‌های هوشمند
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_iou_metric', mode='max'),
        EarlyStopping(patience=20, monitor='val_iou_metric', mode='max'),
        ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_iou_metric', mode='max')
    ]
    
    # آموزش
    history = model.fit(
        train_ds,
        validation_data=val_ds,  # نیاز به دیتاست اعتبارسنجی دارید
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

# 5. ارزیابی و پیش‌بینی
def visualize_results(model, dataset, num_samples=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(images[i])
            plt.title('Input Image')
            
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(masks[i,...,0], cmap='gray')
            plt.title('Ground Truth')
            
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(preds[i,...,0] > 0.5, cmap='gray')
            plt.title('Prediction')
        plt.tight_layout()
        plt.show()

# اجرای اصلی
if __name__ == "__main__":
    # مسیرهای داده
    image_dir = "path/to/train_images"
    mask_dir = "path/to/train_masks"
    val_image_dir = "path/to/val_images"
    val_mask_dir = "path/to/val_masks"
    
    # ایجاد دیتاست‌ها
    train_ds = create_dataset(image_dir, mask_dir)
    val_ds = create_dataset(val_image_dir, val_mask_dir, augment=False)
    
    # آموزش مدل
    model, history = train_model()
    
    # نمایش نتایج
    visualize_results(model, val_ds)
