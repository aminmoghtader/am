import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. بهبود دیتالودر با آگمنتاسیون پیشرفته
def advanced_augment(image, mask):
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
    
    # Random brightness/contrast
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    
    return image, mask

# 2. ساخت مدل HRNet پیشرفته با مکانیزم توجه
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    return ReLU()(x)

def channel_attention(x, ratio=8):
    channel = x.shape[-1]
    shared_layer = models.Sequential([
        layers.Dense(channel//ratio, activation='relu'),
        layers.Dense(channel, activation='sigmoid')
    ])
    avg_pool = layers.GlobalAvgPool2D()(x)
    max_pool = layers.GlobalMaxPool2D()(x)
    avg_out = shared_layer(avg_pool)
    max_out = shared_layer(max_pool)
    attention = Add()([avg_out, max_out])
    attention = layers.Reshape((1, 1, channel))(attention)
    return layers.Multiply()([x, attention])

def hrnet_module(x, filters_list):
    # Branch 1 (High-Res)
    b1 = residual_block(x, filters_list[0])
    b1 = channel_attention(b1)
    
    # Branch 2 (Low-Res)
    b2 = layers.AvgPool2D(pool_size=2)(x)
    b2 = residual_block(b2, filters_list[1])
    b2 = channel_attention(b2)
    
    return b1, b2

def build_advanced_hrnet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    
    # Stem
    x = Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Stage 1
    b1, b2 = hrnet_module(x, [64, 128])
    
    # Fusion
    b2_up = UpSampling2D(size=2, interpolation='bilinear')(b2)
    fused = Concatenate()([b1, b2_up])
    
    # Stage 2
    b1, b2 = hrnet_module(fused, [128, 256])
    
    # Final Head
    x = Conv2D(256, 3, padding='same')(fused)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Multi-scale Feature Fusion
    b2_up = UpSampling2D(size=4, interpolation='bilinear')(b2)
    x = Concatenate()([x, b1, b2_up])
    
    # Output
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    
    return Model(inputs, outputs)

# 3. آموزش با تنظیمات پیشرفته
def train_model():
    model = build_advanced_hrnet()
    
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
    
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
                        'accuracy'])
    
    callbacks = [
        ModelCheckpoint('best_hrnet.h5', save_best_only=True, monitor='val_binary_io_u', mode='max'),
        EarlyStopping(patience=15, monitor='val_binary_io_u', mode='max'),
        ReduceLROnPlateau(factor=0.1, patience=5)
    ]
    
    history = model.fit(train_ds, 
                       validation_data=val_ds,
                       epochs=100,
                       callbacks=callbacks)
    
    return model, history

# 4. ارزیابی با معیارهای حرفه‌ای
def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds)
    print(f"IoU: {results[1]:.4f}, Accuracy: {results[2]:.4f}")
    
    # Visualization
    for images, masks in test_ds.take(1):
        preds = model.predict(images)
        plt.figure(figsize=(10, 5))
        for i in range(3):
            plt.subplot(3, 3, i*3+1)
            plt.imshow(images[i])
            plt.title('Input')
            plt.subplot(3, 3, i*3+2)
            plt.imshow(masks[i,...,0], cmap='gray')
            plt.title('Ground Truth')
            plt.subplot(3, 3, i*3+3)
            plt.imshow(preds[i,...,0] > 0.5, cmap='gray')
            plt.title('Prediction')
        plt.tight_layout()
        plt.show()
