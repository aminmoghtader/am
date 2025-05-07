import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# پارامترها
IMG_SIZE = (256, 256)
test_dir = os.path.expanduser("~/Documents/AI/anaconda3/envs/ML/data/img1/test_img/")
mask_dir = os.path.expanduser("~/Documents/AI/anaconda3/envs/ML/data/img1/test_lab/")
test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")])

# بارگذاری مدل
model = tf.keras.models.load_model("unet_plus.h5", compile=False)

# توابع
def load_image_and_mask(img_path, mask_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def iou_coef(y_true, y_pred):
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0.0

def dice_coef(y_true, y_pred):
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0.0

def show_results(image, true_mask, pred_mask, image_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask[:, :, 0], cmap='gray')
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[:, :, 0], cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.suptitle(image_name)
    plt.tight_layout()
    plt.savefig(f"comparison_{image_name}.png")
    plt.show()



# لیست برای ذخیره اطلاعات تصویری برای نمایش بعد از حلقه
comparison_data = []
results = []
for img_path in test_images:
    mask_filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_filename)
    input_img, true_mask = load_image_and_mask(img_path, mask_path)
    pred_mask = model.predict(input_img)[0]

    # پردازش نهایی
    pred_bin = (pred_mask[:, :, 0] > 0.5).astype(np.uint8)
    true_bin = (true_mask[:, :, 0] > 0.5).numpy().astype(np.uint8)

    acc = accuracy_score(true_bin.flatten(), pred_bin.flatten())
    iou = iou_coef(true_bin, pred_bin)
    dice = dice_coef(true_bin, pred_bin)

    results.append({
        "Image": os.path.basename(img_path),
        "Accuracy": acc,
        "IoU": iou,
        "Dice": dice
    })

    # ذخیره ماسک پیش‌بینی‌شده
    os.makedirs("predicted_masks", exist_ok=True)
    tf.keras.utils.save_img(os.path.join("predicted_masks", os.path.basename(img_path)), pred_mask)

    # ذخیره برای نمایش نهایی
    comparison_data.append({
        "image": input_img,
        "true_mask": true_mask.numpy(),
        "pred_mask": np.expand_dims(pred_bin, axis=-1),
        "name": os.path.basename(img_path),
        "acc": acc,
        "iou": iou,
        "dice": dice
    })

# ساخت فایل اکسل
df = pd.DataFrame(results)
df.to_excel("segmentation_metrics.xlsx", index=False)

# نمایش همه نتایج تصویری بعد از حلقه
for item in comparison_data:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(item["image"][0])
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(item["true_mask"][:, :, 0], cmap='gray')
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(item["pred_mask"][:, :, 0], cmap='gray')
    plt.title(f"Predicted\nAcc: {item['acc']:.2f}, IoU: {item['iou']:.2f}, Dice: {item['dice']:.2f}")
    plt.axis('off')

    plt.suptitle(item["name"])
    plt.tight_layout()
    plt.savefig(f"comparison_{item['name']}")
    plt.show()

# رسم نمودار
plt.figure(figsize=(12, 5))
plt.plot(df["Image"], df["IoU"], marker='o', label="IoU", color='green')
plt.plot(df["Image"], df["Dice"], marker='s', label="Dice", color='purple')
plt.xticks(rotation=45)
plt.title("IoU and Dice per Image")
plt.xlabel("Image")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("iou_dice_plot.png")
plt.show()
