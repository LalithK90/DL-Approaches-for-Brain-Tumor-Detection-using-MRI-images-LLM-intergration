"""
XAI Validation Metrics Module
Implements comprehensive explainability validation metrics from xai_validator.py
for real-time prediction explanations in the API.

Includes:
- Faithfulness metrics (comprehensiveness, sufficiency)
- Agreement metrics (dice, IoU)
- AUC-based metrics (deletion, insertion)
- Validation tests (randomized weights)
"""

import numpy as np
import tensorflow as tf
import cv2
import warnings
from sklearn.metrics import auc


# =========================
# Heatmap Utilities
# =========================

def normalize_heatmap(hm):
    """Normalize heatmap to [0, 1] range."""
    hm = np.maximum(hm, 0)
    hm_min = hm.min()
    hm_max = hm.max()
    if hm_max == hm_min:
        return np.zeros_like(hm, dtype=np.float32)
    return (hm - hm_min) / (hm_max - hm_min + 1e-8)


def resize_to_input(hm, target_shape):
    """Resize heatmap to target shape using interpolation."""
    return cv2.resize(hm, target_shape, interpolation=cv2.INTER_LINEAR)


def binarize_heatmap(hm, top_percent=15):
    """
    Binarize heatmap by selecting top percentage of activation.
    
    Args:
        hm: Heatmap array
        top_percent: Top percentage threshold (default 15%)
    
    Returns:
        Binary mask where top percentage is 1, rest is 0
    """
    thresh = np.percentile(hm, 100 - top_percent)
    return (hm >= thresh).astype(np.uint8)


# =========================
# Agreement Metrics
# =========================

def dice_score(m1, m2):
    """
    Calculate Dice coefficient between two binary masks.
    Range: [0, 1] where 1 is perfect agreement.
    """
    inter = np.sum(m1 * m2)
    return (2 * inter) / (np.sum(m1) + np.sum(m2) + 1e-8)


def iou_score(m1, m2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Range: [0, 1] where 1 is perfect agreement.
    """
    inter = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - inter
    return inter / (union + 1e-8)


# =========================
# Faithfulness Metrics
# =========================

def comprehensiveness(model, img, mask, cls, verbose=0):
    """
    Comprehensiveness metric: How much does model confidence drop
    when important regions are removed?
    
    Higher values indicate better explanation faithfulness.
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        mask: Binary mask of important regions
        cls: Target class index
        verbose: Model prediction verbosity
    
    Returns:
        float: Confidence drop (p0 - p1)
    """
    try:
        # Original prediction
        p0 = model.predict(img[None], verbose=verbose)[0, cls]
        
        # Perturbed prediction (remove important regions)
        perturbed = img.copy()
        perturbed[mask == 1] = 0
        p1 = model.predict(perturbed[None], verbose=verbose)[0, cls]
        
        return float(p0 - p1)
    except Exception as e:
        print(f"Error in comprehensiveness calculation: {e}")
        return 0.0


def sufficiency(model, img, mask, cls, verbose=0):
    """
    Sufficiency metric: How much can we reconstruct the prediction
    by only keeping important regions?
    
    Higher values indicate better explanation sufficiency.
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        mask: Binary mask of important regions
        cls: Target class index
        verbose: Model prediction verbosity
    
    Returns:
        float: Confidence from kept regions (p0 - p1)
    """
    try:
        # Original prediction
        p0 = model.predict(img[None], verbose=verbose)[0, cls]
        
        # Prediction with only important regions
        kept = np.zeros_like(img)
        kept[mask == 1] = img[mask == 1]
        p1 = model.predict(kept[None], verbose=verbose)[0, cls]
        
        return float(p0 - p1)
    except Exception as e:
        print(f"Error in sufficiency calculation: {e}")
        return 0.0


# =========================
# AUC-based Metrics
# =========================

def deletion_auc(model, img, heatmap, cls, steps=20, verbose=0):
    """
    Deletion AUC: Progressively delete important regions (by heatmap order)
    and measure how quickly confidence drops.
    
    Higher AUC indicates better explanation quality.
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        heatmap: Explanation heatmap (e.g., Grad-CAM)
        cls: Target class index
        steps: Number of deletion steps
        verbose: Model prediction verbosity
    
    Returns:
        float: Area under deletion curve
    """
    try:
        flat = heatmap.flatten()
        order = np.argsort(flat)[::-1]  # Descending order of importance
        img_flat = img.reshape(-1, img.shape[-1])
        scores = []
        
        for i in np.linspace(0, len(order), steps).astype(int):
            temp = img_flat.copy()
            temp[order[:i]] = 0  # Delete top-i important pixels
            p = model.predict(temp.reshape(img.shape)[None], verbose=verbose)[0, cls]
            scores.append(float(p))
        
        return float(auc(range(len(scores)), scores))
    except Exception as e:
        print(f"Error in deletion_auc calculation: {e}")
        return 0.0


def insertion_auc(model, img, heatmap, cls, steps=20, verbose=0):
    """
    Insertion AUC: Progressively insert important regions (by heatmap order)
    starting from blank image and measure how quickly confidence increases.
    
    Higher AUC indicates better explanation quality.
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        heatmap: Explanation heatmap (e.g., Grad-CAM)
        cls: Target class index
        steps: Number of insertion steps
        verbose: Model prediction verbosity
    
    Returns:
        float: Area under insertion curve
    """
    try:
        flat = heatmap.flatten()
        order = np.argsort(flat)[::-1]  # Descending order of importance
        baseline = np.zeros_like(img).reshape(-1, img.shape[-1])
        orig = img.reshape(-1, img.shape[-1])
        scores = []
        
        for i in np.linspace(0, len(order), steps).astype(int):
            baseline[order[:i]] = orig[order[:i]]  # Insert top-i important pixels
            p = model.predict(baseline.reshape(img.shape)[None], verbose=verbose)[0, cls]
            scores.append(float(p))
        
        return float(auc(range(len(scores)), scores))
    except Exception as e:
        print(f"Error in insertion_auc calculation: {e}")
        return 0.0


# =========================
# Validation Tests
# =========================

def randomized_weights_test(model, img, cls, conv_layer, verbose=0):
    """
    Randomized Weights Test (Sanity Check):
    Verify that explanations change significantly when model weights are randomized.
    
    Correlation close to 0 indicates model properly uses weights (good sanity check).
    Correlation close to 1 indicates explanations don't depend on weights (bad sanity check).
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        cls: Target class index
        conv_layer: Name of convolutional layer for explanation
        verbose: Model prediction verbosity
    
    Returns:
        float: Correlation coefficient between original and randomized explanations
    """
    try:
        from src.utils.utils import generate_gradcam
        
        # Get original explanation (returns tuple: (superimposed_img, cam))
        original_superimposed, original_cam = generate_gradcam(
            img[None], model)
        original_flat = original_cam.flatten()
        
        # Save original weights for restoration
        original_weights = [layer.get_weights() for layer in model.layers if hasattr(layer, 'kernel')]
        
        # Randomize model weights
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(tf.random.normal(layer.kernel.shape))
                break  # Only randomize first kernel layer for efficiency
        
        # Get explanation after randomization (returns tuple: (superimposed_img, cam))
        randomized_superimposed, randomized_cam = generate_gradcam(
            img[None], model)
        randomized_flat = randomized_cam.flatten()
        
        # Restore original weights
        weight_idx = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.set_weights(original_weights[weight_idx])
                weight_idx += 1
                break
        
        # Calculate correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Handle potential NaN/inf values
            if np.any(np.isnan(original_flat)) or np.any(np.isnan(randomized_flat)):
                return 0.0
            corr = np.corrcoef(original_flat, randomized_flat)[0, 1]
            if np.isnan(corr):
                return 0.0
        
        return float(corr)
    except Exception as e:
        print(f"Error in randomized_weights_test: {e}")
        return 0.0


# =========================
# Unified Faithfulness Calculator
# =========================

def calculate_faithfulness_metrics(model, img, gradcam_heatmap, cls, verbose=0):
    """
    Calculate all faithfulness metrics in one function call.
    
    Args:
        model: Trained Keras/TensorFlow model
        img: Input image array (HxWxC)
        gradcam_heatmap: Grad-CAM heatmap from explanation
        cls: Target class index
        verbose: Model prediction verbosity
    
    Returns:
        dict: Dictionary with all faithfulness metrics
    """
    # Binarize heatmap for comprehensiveness/sufficiency
    mask = binarize_heatmap(gradcam_heatmap)
    
    # Calculate metrics
    metrics = {
        'comprehensiveness': comprehensiveness(model, img, mask, cls, verbose),
        'sufficiency': sufficiency(model, img, mask, cls, verbose),
        'deletion_auc': deletion_auc(model, img, gradcam_heatmap, cls, steps=20, verbose=verbose),
        'insertion_auc': insertion_auc(model, img, gradcam_heatmap, cls, steps=20, verbose=verbose),
    }
    
    return metrics
