import os
import sys
import time
import csv
import platform
import subprocess
import numpy as np
import tensorflow as tf
from datetime import datetime

# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model and utility functions
from brain_tumor_identification_api.src.models.models import MODELS, IMAGE_SIZE
from brain_tumor_identification_api.src.utils.utils import read_image, z_score_per_image, brier_score, softmax_entropy, margin_score, mc_dropout_predict

TESTING_FOLDER = os.path.join(PROJECT_ROOT, 'XAI_validation/testing')
NUM_IMAGES_PER_CLASS = 10
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MC_DROPOUT_SAMPLES = 10
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def get_gpu_info():
    """Return a string describing available GPUs (CUDA or Apple Silicon)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return ', '.join(d.name for d in gpus)
    # Try nvidia-smi for CUDA systems
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 'None detected'


def get_ram_info():
    """Return total and available RAM in GB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"Total: {vm.total / 1e9:.2f} GB, Available: {vm.available / 1e9:.2f} GB"
    except ImportError:
        return 'psutil not installed — install with: pip install psutil'


def get_cpu_info():
    """Return CPU model string."""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        logical_count = psutil.cpu_count(logical=True)
        cpu_str = f"{cpu_count} physical cores, {logical_count} logical cores"
    except ImportError:
        cpu_str = f"{os.cpu_count()} logical cores"

    # Try to get CPU brand string
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"{result.stdout.strip()} — {cpu_str}"
        elif platform.system() == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        brand = line.split(':')[1].strip()
                        return f"{brand} — {cpu_str}"
    except Exception:
        pass
    return f"{platform.processor() or 'Unknown CPU'} — {cpu_str}"


def get_opencv_version():
    """Return OpenCV version or indicate it is not installed."""
    try:
        import cv2
        return cv2.__version__
    except ImportError:
        return 'Not installed'


def save_experimental_setup(output_path):
    """Collect and write system/library info to a .txt file."""
    lines = [
        "=" * 60,
        "EXPERIMENTAL SETUP",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "--- Operating System ---",
        f"OS:            {platform.system()} {platform.release()}",
        f"OS Version:    {platform.version()}",
        f"Machine:       {platform.machine()}",
        f"Node:          {platform.node()}",
        "",
        "--- CPU ---",
        f"CPU:           {get_cpu_info()}",
        "",
        "--- RAM ---",
        f"RAM:           {get_ram_info()}",
        "",
        "--- GPU ---",
        f"GPU:           {get_gpu_info()}",
        "",
        "--- Python ---",
        f"Python:        {platform.python_version()} ({sys.executable})",
        "",
        "--- Key Libraries ---",
        f"TensorFlow:    {tf.__version__}",
        f"NumPy:         {np.__version__}",
        f"OpenCV:        {get_opencv_version()}",
        "",
        "--- Experiment Parameters ---",
        f"Testing folder:         {TESTING_FOLDER}",
        f"Classes:                {CLASSES}",
        f"Images per class:       {NUM_IMAGES_PER_CLASS}",
        f"Total test images:      {len(CLASSES) * NUM_IMAGES_PER_CLASS}",
        f"Image size:             {IMAGE_SIZE}",
        f"MC Dropout samples (T): {MC_DROPOUT_SAMPLES}",
        "=" * 60,
        "",
    ]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Experimental setup saved to: {output_path}")


def load_test_images():
    """Load and preprocess test images from each class folder."""
    images = []
    labels = []
    for idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(TESTING_FOLDER, class_name)
        if not os.path.isdir(class_folder):
            raise FileNotFoundError(f"Class folder not found: {class_folder}")
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        selected = image_files[:NUM_IMAGES_PER_CLASS]
        if len(selected) < NUM_IMAGES_PER_CLASS:
            print(f"Warning: only {len(selected)} images found for class '{class_name}' (expected {NUM_IMAGES_PER_CLASS})")
        for img_file in selected:
            img_path = os.path.join(class_folder, img_file)
            img = read_image(IMAGE_SIZE, img_path)
            images.append(img)
            label = np.zeros(len(CLASSES))
            label[idx] = 1
            labels.append(label)
    if not images:
        raise ValueError("No images loaded. Check TESTING_FOLDER and class subfolders.")
    images = np.array(images)
    labels = np.array(labels)
    images = z_score_per_image(images)
    return images, labels


def main():
    """Run efficiency evaluation for all models and save results to CSV."""
    results_csv = f"model_prediction_efficiency_{TIMESTAMP}.csv"
    setup_txt = f"experimental_setup_{TIMESTAMP}.txt"

    save_experimental_setup(setup_txt)

    images, labels = load_test_images()
    print(f"Loaded {len(images)} images for testing.")

    fieldnames = ['model', 'total_time', 'avg_time', 'accuracy', 'brier_score', 'softmax_entropy', 'margin', 'mc_dropout_var']
    with open(results_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating model: {model_name}")
        try:
            model = tf.keras.models.load_model(
                os.path.join(PROJECT_ROOT, 'brain_tumor_identification_api', model_path),
                compile=False
            )
        except OSError as e:
            print(f"  Skipping {model_name}: could not load model — {e}")
            continue

        total_time = 0.0
        correct = 0
        brier_scores = []
        entropies = []
        margins = []
        mc_vars = []

        try:
            input_name = model.inputs[0].name.split(':')[0]
        except (AttributeError, IndexError):
            input_name = None

        for img, label in zip(images, labels):
            img_batch = np.expand_dims(img, axis=0)
            model_input = {input_name: img_batch} if input_name else img_batch
            start = time.time()
            pred = model.predict(model_input, verbose=0)[0]
            elapsed = time.time() - start
            total_time += elapsed

            pred_class = np.argmax(pred)
            true_class = np.argmax(label)
            if pred_class == true_class:
                correct += 1
            brier_scores.append(brier_score(pred, true_class))
            entropies.append(softmax_entropy(pred))
            margins.append(margin_score(pred))
            _, mc_var, _, _ = mc_dropout_predict(model, model_input, T=MC_DROPOUT_SAMPLES)
            mc_vars.append(np.mean(mc_var))

        avg_time = total_time / len(images)
        accuracy = correct / len(images)
        result_row = {
            'model': model_name,
            'total_time': round(total_time, 4),
            'avg_time': round(avg_time, 6),
            'accuracy': round(accuracy, 4),
            'brier_score': round(float(np.mean(brier_scores)), 4),
            'softmax_entropy': round(float(np.mean(entropies)), 4),
            'margin': round(float(np.mean(margins)), 4),
            'mc_dropout_var': round(float(np.mean(mc_vars)), 6),
        }
        print(f"  Total: {total_time:.2f}s | Avg: {avg_time:.4f}s | Acc: {accuracy:.3f}")

        with open(results_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result_row)

    print(f"\nResults saved to: {results_csv}")
    print(f"Setup info saved to: {setup_txt}")


if __name__ == "__main__":
    main()
