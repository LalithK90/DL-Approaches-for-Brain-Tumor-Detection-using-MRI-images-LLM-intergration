import numpy as np
import tensorflow as tf
import cv2
import os
import random
import sys
from contextlib import contextmanager
from sklearn.metrics import auc
from lime import lime_image
import keras
from datetime import datetime
import warnings


# Context manager to suppress output
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# =========================
# Utility functions
# =========================


def read_image(image_size, image_path):
    """Read and preprocess image using the same method as the API"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_AREA)
    array = keras.utils.img_to_array(img)
    return array


def z_score_per_image(img_array):
    """Normalize image array using z-score normalization"""
    mean = np.mean(img_array, axis=(1, 2, 3), keepdims=True)
    std = np.std(img_array, axis=(1, 2, 3), keepdims=True)
    return (img_array - mean) / (std + 1e-8)


def load_random_images_from_testing(testing_folder, num_images_per_class=10, image_size=224):
    """
    Load random images from testing folder with equal distribution across classes.
    
    Args:
        testing_folder: Path to testing folder containing class subdirectories
        num_images_per_class: Number of random images to load per class
        image_size: Target image size for model input
    
    Returns:
        images: Array of preprocessed images
        labels: Array of one-hot encoded labels
        class_names: List of class names
    """
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    images = []
    labels = []
    
    for class_name in class_names:
        class_folder = os.path.join(testing_folder, class_name)
        
        if not os.path.exists(class_folder):
            print(f"Warning: {class_folder} does not exist, skipping...")
            continue
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Randomly select images
        selected_files = random.sample(image_files, min(num_images_per_class, len(image_files)))
        
        print(f"Loading {len(selected_files)} images from {class_name}...")
        
        for img_file in selected_files:
            img_path = os.path.join(class_folder, img_file)
            try:
                img_array = read_image(image_size, img_path)
                images.append(img_array)
                
                # Create one-hot encoded label
                label = np.zeros(len(class_names))
                label[label_map[class_name]] = 1
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Check if any images were loaded
    if len(images) == 0:
        raise ValueError("No images were loaded. Please check the testing folder path and contents.")
    
    # Apply z-score normalization
    images = z_score_per_image(images)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return images, labels, class_names


def normalize_heatmap(hm):
    hm = np.maximum(hm, 0)
    return (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)


def resize_to_input(hm, target_shape):
    return cv2.resize(hm, target_shape, interpolation=cv2.INTER_LINEAR)


def binarize_heatmap(hm, top_percent=15):
    thresh = np.percentile(hm, 100 - top_percent)
    return (hm >= thresh).astype(np.uint8)


def dice_score(m1, m2):
    inter = np.sum(m1 * m2)
    return (2 * inter) / (np.sum(m1) + np.sum(m2) + 1e-8)


def iou_score(m1, m2):
    inter = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - inter
    return inter / (union + 1e-8)

# =========================
# Explanation generators
# =========================


def generate_gradcam(model, image, class_idx, conv_layer):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image[None])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)

    cam = normalize_heatmap(cam.numpy())
    cam = resize_to_input(cam, image.shape[:2])  # 🔥 CRITICAL FIX
    return cam


def generate_saliency(model, image, class_idx):
    image_tf = tf.convert_to_tensor(image[None])
    with tf.GradientTape() as tape:
        tape.watch(image_tf)
        preds = model(image_tf)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, image_tf)[0]
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)
    return normalize_heatmap(saliency.numpy())


def generate_lime_heatmap(model, image, samples=1000):
    explainer = lime_image.LimeImageExplainer()
    
    # Suppress LIME's verbose output
    with suppress_output():
        explanation = explainer.explain_instance(
            image.astype(np.double),
            classifier_fn=lambda x: model.predict(x, verbose=0),
            top_labels=1,
            hide_color=0,
            num_samples=samples
        )

    _, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False
    )

    return normalize_heatmap(mask.astype(float))

# =========================
# Faithfulness metrics
# =========================


def comprehensiveness(model, img, mask, cls):
    p0 = model.predict(img[None], verbose=0)[0, cls]
    perturbed = img.copy()
    perturbed[mask == 1] = 0
    p1 = model.predict(perturbed[None], verbose=0)[0, cls]
    return p0 - p1


def sufficiency(model, img, mask, cls):
    p0 = model.predict(img[None], verbose=0)[0, cls]
    kept = np.zeros_like(img)
    kept[mask == 1] = img[mask == 1]
    p1 = model.predict(kept[None], verbose=0)[0, cls]
    return p0 - p1


def deletion_auc(model, img, heatmap, cls, steps=20):
    flat = heatmap.flatten()
    order = np.argsort(flat)[::-1]
    img_flat = img.reshape(-1, img.shape[-1])
    scores = []

    for i in np.linspace(0, len(order), steps).astype(int):
        temp = img_flat.copy()
        temp[order[:i]] = 0
        p = model.predict(temp.reshape(img.shape)[None], verbose=0)[0, cls]
        scores.append(p)

    return auc(range(len(scores)), scores)


def insertion_auc(model, img, heatmap, cls, steps=20):
    flat = heatmap.flatten()
    order = np.argsort(flat)[::-1]
    baseline = np.zeros_like(img).reshape(-1, img.shape[-1])
    orig = img.reshape(-1, img.shape[-1])
    scores = []

    for i in np.linspace(0, len(order), steps).astype(int):
        baseline[order[:i]] = orig[order[:i]]
        p = model.predict(baseline.reshape(img.shape)[None], verbose=0)[0, cls]
        scores.append(p)

    return auc(range(len(scores)), scores)

# =========================
# Uncertainty & calibration
# =========================


def softmax_entropy(p):
    return -np.sum(p * np.log(p + 1e-8))


def prediction_margin(p):
    top2 = np.sort(p)[-2:]
    return top2[1] - top2[0]


def brier_score(p, y_true):
    if isinstance(y_true, (list, np.ndarray)):
        y_true = int(np.argmax(y_true))

    y = np.zeros_like(p)
    y[y_true] = 1
    return np.sum((p - y) ** 2)


def mc_dropout(model, img, T=30):
    preds = np.array([
        model(img[None], training=True).numpy()[0]
        for _ in range(T)
    ])
    return preds.mean(axis=0), preds.var(axis=0)

# =========================
# Sanity check
# =========================


def randomized_weights_test(model, img, conv_layer):
    original = generate_gradcam(model, img, 0, conv_layer)

    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel.assign(tf.random.normal(layer.kernel.shape))
            break

    randomized = generate_gradcam(model, img, 0, conv_layer)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(original.flatten(), randomized.flatten())[0, 1]
    
    return corr


# =========================
# MAIN VALIDATION FUNCTION
# =========================
def xai_validate_model(model, images, labels, conv_layer, max_samples=30, title="XAI Metrics", output_file=None):
    results = []

    for img, y in zip(images[:max_samples], labels[:max_samples]):
        probs = model.predict(img[None], verbose=0)[0]
        cls = probs.argmax()

        # -------- Explanations --------
        gc = generate_gradcam(model, img, cls, conv_layer)
        sm = generate_saliency(model, img, cls)
        lm = generate_lime_heatmap(model, img)

        gc_m = binarize_heatmap(gc)
        sm_m = binarize_heatmap(sm)
        lm_m = binarize_heatmap(lm)

        top3 = np.sort(probs)[-3:]

        results.append([
            dice_score(gc_m, lm_m),        # Grad-CAM vs LIME Dice
            iou_score(gc_m, lm_m),         # Grad-CAM vs LIME IoU
            dice_score(gc_m, sm_m),        # Grad-CAM vs Saliency Dice
            iou_score(gc_m, sm_m),         # Grad-CAM vs Saliency IoU
            comprehensiveness(model, img, gc_m, cls),
            sufficiency(model, img, gc_m, cls),
            deletion_auc(model, img, gc, cls),
            insertion_auc(model, img, gc, cls),
            brier_score(probs, y),
            softmax_entropy(probs),
            prediction_margin(probs),
            mc_dropout(model, img)[1].mean(),
            top3[0],                       # Top-3 (lowest)
            top3[1],                       # Top-3 (middle)
            top3[2],                       # Top-3 (highest)
            randomized_weights_test(model, img, conv_layer)
        ])

    mean_metrics = np.nanmean(results, axis=0)

    headers = [
        "Dice (GC vs LIME)",
        "IoU (GC vs LIME)",
        "Dice (GC vs Saliency)",
        "IoU (GC vs Saliency)",
        "Comprehensiveness",
        "Sufficiency",
        "Deletion AUC",
        "Insertion AUC",
        "Brier Score",
        "Softmax Entropy",
        "Prediction Margin",
        "MC Dropout Variance",
        "Top-3 Prob (3rd)",
        "Top-3 Prob (2nd)",
        "Top-3 Prob (1st)",
        "Randomized Weights Corr."
    ]


    # Format output
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"{title}")
    output_lines.append(f"{'='*80}")
    
    for header, value in zip(headers, mean_metrics):
        output_lines.append(f"{header:<30}: {value:.6f}")
    output_lines.append(f"{'='*80}\n")
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'a') as f:
            for line in output_lines:
                f.write(line + '\n')


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configuration
    TESTING_FOLDER = "testing"  # Path to testing folder
    MODEL_PATHS = [
        "../brain_tumor_identification_api/models/propose_balance.h5",
        "../brain_tumor_identification_api/models/propose_imbalanced.h5"
    ]
    IMAGE_SIZE = 224
    NUM_IMAGES_PER_CLASS = 20  # Number of random images per class
    MAX_SAMPLES = 80  # Maximum samples for validation (20 per class * 4 classes)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"xai_validation_results_{timestamp}.txt"
    
    # Initialize output file
    with open(output_filename, 'w') as f:
        f.write("=" * 80 + '\n')
        f.write("XAI MODEL VALIDATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + '\n')
        f.write(f"Number of images per class: {NUM_IMAGES_PER_CLASS}\n")
        f.write(f"Total samples: {MAX_SAMPLES}\n")
        f.write(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write("=" * 80 + '\n\n')
    
    print("=" * 80)
    print("XAI MODEL VALIDATION")
    print("=" * 80)
    print(f"Results will be saved to: {output_filename}")
    
    # Load random images from testing folder (only once, reused for both models)
    print("\n[1/4] Loading random images from testing folder...")
    images, labels, class_names = load_random_images_from_testing(
        TESTING_FOLDER, 
        num_images_per_class=NUM_IMAGES_PER_CLASS,
        image_size=IMAGE_SIZE
    )
    
    print(f"\nClass names: {class_names}")
    
    # Test both models
    for model_idx, model_path in enumerate(MODEL_PATHS, 1):
        print("\n" + "=" * 80)
        print(f"TESTING MODEL {model_idx}/{len(MODEL_PATHS)}: {model_path}")
        print("=" * 80)
        
        # Load model
        print(f"\n[{model_idx*2}/4] Loading model...")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model_path}")
        
        # Get last convolutional layer
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer = layer.name
                break
        
        if not conv_layer:
            raise ValueError("No convolutional layer found in the model")
        
        print(f"Using convolutional layer: {conv_layer}")
        
        # Run XAI validation
        print(f"\n[{model_idx*2+1}/4] Running XAI validation...")
        model_name = model_path.split('/')[-1].replace('.h5', '')
        xai_validate_model(
            model=model,
            images=images,
            labels=labels,
            conv_layer=conv_layer,
            max_samples=min(MAX_SAMPLES, len(images)),
            title=f"XAI Validation Results - {model_name.upper()}",
            output_file=output_filename
        )
        
        # Clear model from memory
        del model
        tf.keras.backend.clear_session()
    
    # Write completion message to file
    with open(output_filename, 'a') as f:
        f.write("\n" + "=" * 80 + '\n')
        f.write("VALIDATION COMPLETE FOR ALL MODELS\n")
        f.write("=" * 80 + '\n')
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE FOR ALL MODELS")
    print("=" * 80)
    print(f"\nResults saved to: {output_filename}")
