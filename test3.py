import os
import tensorflow as tf
import matplotlib.pyplot as plt

# اندازه تصویر
IMG_SIZE = (256, 256)

# مسیر یک تصویر
img_path = os.path.expanduser("~/Documents/AI/test_images/image1.png")  # مسیر دقیق تصویر

# بارگذاری مدل
model = tf.keras.models.load_model("unet_plus.h5", compile=False)

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0), image  # هم تصویر برای مدل، هم برای نمایش

# پیش‌پردازش
input_img, display_img = preprocess_image(img_path)
pred_mask = model.predict(input_img)[0]

# نمایش نتیجه
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(display_img / 1.0)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask[:, :, 0], cmap='gray')
plt.axis('off')

plt.suptitle(os.path.basename(img_path))
plt.show()

# ذخیره خروجی
os.makedirs("predicted_masks", exist_ok=True)
save_path = os.path.join("predicted_masks", os.path.basename(img_path))
tf.keras.utils.save_img(save_path, pred_mask)
