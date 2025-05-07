

def conv_block(x, filters, dropout_rate=0.1):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_unet_plus_advanced(img_size=(256, 256, 3), dropout_rate=0.1):
    inputs = layers.Input(shape=img_size)

    # Encoder
    x1 = conv_block(inputs, 32, dropout_rate)
    p1 = layers.MaxPooling2D()(x1)

    x2 = conv_block(p1, 64, dropout_rate)
    p2 = layers.MaxPooling2D()(x2)

    x3 = conv_block(p2, 128, dropout_rate)
    p3 = layers.MaxPooling2D()(x3)

    x4 = conv_block(p3, 256, dropout_rate)
    p4 = layers.MaxPooling2D()(x4)

    x5 = conv_block(p4, 512, dropout_rate)  # bottleneck

    # Decoder
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x5)
    u4 = layers.concatenate([u4, x4])
    x6 = conv_block(u4, 256, dropout_rate)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x6)
    u3 = layers.concatenate([u3, x3])
    x7 = conv_block(u3, 128, dropout_rate)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(x7)
    u2 = layers.concatenate([u2, x2])
    x8 = conv_block(u2, 64, dropout_rate)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x8)
    u1 = layers.concatenate([u1, x1])
    x9 = conv_block(u1, 32, dropout_rate)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x9)

    return models.Model(inputs, outputs)
