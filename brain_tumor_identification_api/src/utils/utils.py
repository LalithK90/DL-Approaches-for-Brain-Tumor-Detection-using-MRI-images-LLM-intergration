import cv2
import numpy as np
import os
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.stats import entropy
from tf_explain.core.grad_cam import GradCAM

VISUALIZATION_FOLDER = 'static/visualizations'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(image_size, image_path):
    read_image_values = []
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_AREA)
    read_image_values.append(img)
    return read_image_values[0]

def z_score_per_image(img_array):
    mean = np.mean(img_array, axis=(1, 2, 3), keepdims=True)
    std = np.std(img_array, axis=(1, 2, 3), keepdims=True)
    return (img_array - mean) / (std + 1e-8)

def get_last_convolutional_layer(model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    return last_conv_layer_name

def generate_gradcam(img_array, model):
    layer_name = get_last_convolutional_layer(model)
    if not layer_name:
        raise ValueError("No convolutional layer found in the model")

    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    cam = gradcam(CategoricalScore(class_idx), img_array,
                  penultimate_layer=layer_name)

    cam = np.squeeze(cam)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    base_img = img_array[0]
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    superimposed_img = heatmap * 0.4 + base_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, cam  # Return cam for Guided GradCAM


def generate_saliency_map(img_array, model):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    saliency = (saliency - np.min(saliency)) / \
        (np.max(saliency) - np.min(saliency))

    saliency = np.uint8(255 * saliency)
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    return saliency


def generate_lime(img_array, model):
    explainer = lime_image.LimeImageExplainer()

    # Wrapper function for prediction to ensure correct batch dimension for Keras
    def predict_fn(images):
        return model.predict(images)

    explanation = explainer.explain_instance(
        img_array[0],
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_img = mark_boundaries(img_array[0]/255.0, mask)
    lime_img = np.uint8(lime_img * 255)
    return lime_img


def analyze_gradcam_heatmap(heatmap, analysis_filename=None):
    # Normalize heatmap to [0,1]
    heatmap = heatmap.astype(np.float32)
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Compute center of mass
    com_y, com_x = center_of_mass(heatmap)

    # Compute distance to image center
    h, w = heatmap.shape
    center_distance = np.sqrt((com_x - w/2)**2 + (com_y - h/2)**2)

    # Compute activation area ratio
    threshold = 0.5
    activated_area = np.sum(heatmap > threshold)
    total_area = heatmap.size
    activation_ratio = activated_area / total_area

    # Visualize heatmap (save to file, do NOT show)
    fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='jet')
    ax.plot(w/2, h/2, 'wo', label='Image Center')
    ax.plot(com_x, com_y, 'ro', label='Heatmap Center of Mass')
    ax.legend()
    ax.set_title("Grad-CAM Heatmap Analysis")
    plt.axis('off')
    if analysis_filename:
        plt.savefig(analysis_filename, bbox_inches='tight')
    plt.close(fig)

    return center_distance, activation_ratio


def dice_iou(map1, map2):
    intersection = np.logical_and(map1, map2).sum()
    dice = 2 * intersection / (map1.sum() + map2.sum() + 1e-8)
    iou = intersection / (np.logical_or(map1, map2).sum() + 1e-8)
    return dice, iou


def get_top3_probs(pred, labels):
    idxs = np.argsort(pred)[::-1][:3]
    return [(labels[i], float(pred[i])) for i in idxs]


def softmax_entropy(pred):
    return float(entropy(pred, base=2))


def margin_score(pred):
    sorted_pred = np.sort(pred)[::-1]
    return float(sorted_pred[0] - sorted_pred[1])


def mc_dropout_predict(model, img_array, T=30):
    # Run model with dropout enabled (training=True)
    # Wrap input in a list to match Keras functional model expectations
    preds = np.array([model([img_array], training=True).numpy()[0]
                     for _ in range(T)])
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    ci_low = np.percentile(preds, 2.5, axis=0)
    ci_high = np.percentile(preds, 97.5, axis=0)
    return mean, var, ci_low, ci_high


def brier_score(pred, true_idx):
    y_true = np.zeros_like(pred)
    y_true[true_idx] = 1
    return float(np.mean((pred - y_true) ** 2))


def generate_gradcam_tf_explain(model, img_array, class_index, layer_name, save_path):
    explainer = GradCAM()
    data = ([img_array], None)
    grid = explainer.explain(data, model, class_index, layer_name)
    plt.imsave(save_path, grid)
    return grid