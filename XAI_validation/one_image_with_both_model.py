import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from glob import glob

# =========================
# CONFIG
# =========================
IMAGE_SIZE = 224
# If you provide a list of paths here, the script will choose one at random.
TEST_IMAGE_PATHS = []  
TEST_IMAGE_PATH = "testing/glioma/Tr-gl_001.jpg"
MODEL_BALANCED_PATH = "../brain_tumor_identification_api/models/propose_balance.h5"
MODEL_IMBALANCED_PATH = "../brain_tumor_identification_api/models/propose_imbalanced.h5"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# =========================
# IMAGE PREPROCESSING
# =========================


def read_image(image_path, image_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image (cv2.imread returned None): {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)

    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)

    mean = img.mean()
    std = img.std() + 1e-8
    img = (img - mean) / std

    return img


def read_raw_image(image_path, image_size):
    """Read the original image without preprocessing, for display purpose."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image (cv2.imread returned None): {image_path}")

    img = cv2.resize(img, (image_size, image_size))
    # convert grayscale to 3-channel RGB for consistent display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb


def pick_random_image_from_class(class_name: str, root_dir: str) -> str:
    """Pick a random image path from testing/<class_name> under root_dir."""
    class_dir = os.path.join(root_dir, "testing", class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(class_dir, p)))

    if not files:
        raise FileNotFoundError(f"No images found in {class_dir}")

    return random.choice(files)


def list_all_test_images(root_dir: str) -> list:
    """Return a list of all image files under testing/* in root_dir."""
    testing_dir = os.path.join(root_dir, "testing")
    if not os.path.isdir(testing_dir):
        return []
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for cls in os.listdir(testing_dir):
        cls_dir = os.path.join(testing_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for p in patterns:
            files.extend(glob(os.path.join(cls_dir, p)))
    return files


def resolve_image_path(test_image_path: str,
                       default_class: str = "glioma",
                       candidate_paths: list | None = None,
                       random_from_all: bool = True) -> str:
    """Resolve a valid image path.

    Resolution priority:
    1) If candidate_paths provided, choose a random readable one.
    2) Else if test_image_path exists and is readable, use it.
    3) Else if random_from_all, pick random from all testing/*
    4) Else, pick random from testing/<default_class>.
    """
    # Make path relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, test_image_path) if not os.path.isabs(test_image_path) else test_image_path

    # 1) Candidate paths list
    if candidate_paths:
        readable = []
        for p in candidate_paths:
            p_abs = os.path.join(script_dir, p) if not os.path.isabs(p) else p
            if os.path.exists(p_abs):
                tmp = cv2.imread(p_abs, cv2.IMREAD_GRAYSCALE)
                if tmp is not None:
                    readable.append(p_abs)
        if readable:
            return random.choice(readable)

    if os.path.exists(candidate):
        # Verify OpenCV can read it
        tmp = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
        if tmp is not None:
            return candidate

    # 3) Random from all testing/*
    if random_from_all:
        all_imgs = list_all_test_images(script_dir)
        if all_imgs:
            return random.choice(all_imgs)

    # 4) Fallback: pick random image from default class
    return pick_random_image_from_class(default_class, script_dir)

# =========================
# GRAD-CAM
# =========================


def normalize_heatmap(hm):
    hm = np.maximum(hm, 0)
    return (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)


def generate_gradcam(model, image, class_idx, conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image[None])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)

    cam = cam.numpy()
    cam = normalize_heatmap(cam)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    return cam

# =========================
# VISUALIZATION
# =========================


def overlay(image, heatmap):
    # Normalize the base image for display
    base = image.astype(np.float32)
    base_min, base_max = base.min(), base.max()
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)
    plt.imshow(base)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # Load models
    model_bal = tf.keras.models.load_model(MODEL_BALANCED_PATH)
    model_imb = tf.keras.models.load_model(MODEL_IMBALANCED_PATH)

    # Get last convolutional layer
    conv_layer = None
    for layer in reversed(model_bal.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer = layer.name
            break

    if conv_layer is None:
        raise RuntimeError("No Conv2D layer found")

    print(f"Using conv layer: {conv_layer}")

    # Resolve and load image
    img_path = resolve_image_path(
        TEST_IMAGE_PATH,
        default_class="glioma",
        candidate_paths=TEST_IMAGE_PATHS,
        random_from_all=True,
    )
    print(f"Using image: {img_path}")
    raw_img = read_raw_image(img_path, IMAGE_SIZE)
    img = read_image(img_path, IMAGE_SIZE)

    # Predict class (anchor on balanced model)
    probs = model_bal.predict(img[None], verbose=0)[0]
    cls = probs.argmax()
    class_name = CLASS_NAMES[cls]

    # Generate Grad-CAMs
    gc_bal = generate_gradcam(model_bal, img, cls, conv_layer)
    gc_imb = generate_gradcam(model_imb, img, cls, conv_layer)

    # Plot in one row: Original | Preprocessed | Grad-CAM Balanced | Grad-CAM Imbalanced
    plt.figure(figsize=(16, 4))

    # Original image (no preprocessing)
    plt.subplot(1, 4, 1)
    plt.imshow(raw_img)
    plt.title("Original Image")
    plt.axis("off")

    # Preprocessed input image
    plt.subplot(1, 4, 2)
    base = img.astype(np.float32)
    base_min, base_max = base.min(), base.max()
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)
    plt.imshow(base)
    plt.title("Preprocessed Image")
    plt.axis("off")

    # Grad-CAM overlays
    plt.subplot(1, 4, 3)
    overlay(img, gc_bal)
    plt.title("Grad-CAM (Balanced)")

    plt.subplot(1, 4, 4)
    overlay(img, gc_imb)
    plt.title("Grad-CAM (Imbalanced)")

    plt.suptitle(
        f"Grad-CAM Comparison – Predicted Class: {class_name}", fontsize=14)
    plt.tight_layout()
    out_name = "gradcam_line.png"
    plt.savefig(out_name, dpi=300)
    plt.show()

    print(f"Saved: {out_name}")
