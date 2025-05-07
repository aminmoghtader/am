import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# تنظیمات اولیه
IMG_SIZE = (256, 256)
test_dir = os.path.expanduser("~/Documents/AI/anaconda3/envs/ML/data/img1/test_img/")
test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".png")])

# بارگذاری مدل
model = tf.keras.models.load_model("unet_plus.h5", compile=False)
loss_fn = tf.keras.losses.BinaryCrossentropy()
metric_fn = tf.keras.metrics.BinaryAccuracy()

# مسیر ذخیره‌سازی ماسک‌ها
output_dir = "predicted_masks"
os.makedirs(output_dir, exist_ok=True)

results = []

# پیش‌پردازش تصویر
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)

# حلقه پیش‌بینی
for img_path in test_images:
    input_img = preprocess_image(img_path)
    pred_mask = model.predict(input_img)[0]

    # ذخیره ماسک پیش‌بینی‌شده
    save_path = os.path.join(output_dir, os.path.basename(img_path))
    tf.keras.utils.save_img(save_path, pred_mask)

    # ذخیره اطلاعات (بدون ماسک واقعی فعلاً)
    results.append({
        "Image": os.path.basename(img_path),
        "Loss": None,
        "Accuracy": None
    })

    # نمایش تصویر و ماسک
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(tf.squeeze(input_img))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask[:, :, 0], cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ذخیره نتایج در فایل اکسل
df = pd.DataFrame(results)
df.to_excel("prediction_results.xlsx", index=False)

# چون مقدار Loss و Accuracy فعلاً None هستند، نمودار رسم نمی‌شود
if df.get("Loss") is not None and df["Loss"].notnull().any():
    plt.plot(df["Image"], df["Loss"], label="Loss", color='red')
    plt.xticks(rotation=45)
    plt.title("Loss per Image")
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

if df.get("Accuracy") is not None and df["Accuracy"].notnull().any():
    plt.plot(df["Image"], df["Accuracy"], label="Accuracy", color='green')
    plt.xticks(rotation=45)
    plt.title("Accuracy per Image")
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()

