
import os
import sys
import time
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

# Ensure project root is in sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Import model and utility functions
from brain_tumor_identification_api.src.models.models import MODELS, LABELS, IMAGE_SIZE
from brain_tumor_identification_api.src.utils.utils import read_image, z_score_per_image, brier_score, softmax_entropy, margin_score, mc_dropout_predict

TESTING_FOLDER = os.path.join(PROJECT_ROOT, 'XAI_validation/testing')
NUM_IMAGES_PER_CLASS = 10
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
RESULTS_CSV = f"model_prediction_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def load_test_images():
	images = []
	labels = []
	for idx, class_name in enumerate(CLASSES):
		class_folder = os.path.join(TESTING_FOLDER, class_name)
		image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
		selected = image_files[:NUM_IMAGES_PER_CLASS]
		for img_file in selected:
			img_path = os.path.join(class_folder, img_file)
			img = read_image(IMAGE_SIZE, img_path)
			images.append(img)
			label = np.zeros(len(CLASSES))
			label[idx] = 1
			labels.append(label)
	images = np.array(images)
	labels = np.array(labels)
	images = z_score_per_image(images)
	return images, labels

def main():
	images, labels = load_test_images()
	print(f"Loaded {len(images)} images for testing.")
	results = []
	fieldnames = ['model', 'total_time', 'avg_time', 'accuracy', 'brier_score', 'softmax_entropy', 'margin', 'mc_dropout_var']
	# Write header at the start
	with open(RESULTS_CSV, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

	for model_name, model_path in MODELS.items():
		print(f"\nEvaluating model: {model_name}")
		model = tf.keras.models.load_model(
			os.path.join(PROJECT_ROOT, 'brain_tumor_identification_api', model_path),
			compile=False
		)
		# Print model input details for debugging
		print(f"Model input shape: {model.input_shape}, input names: {getattr(model, 'input_names', 'N/A')}")
		total_time = 0.0
		correct = 0
		brier_scores = []
		entropies = []
		margins = []
		mc_vars = []
			for i, (img, label) in enumerate(zip(images, labels)):
			    # Ensure input batch matches model input shape
			    expected_shape = model.input_shape
			    if expected_shape[0] is None:
				batch_shape = (1,) + expected_shape[1:]
			    else:
				batch_shape = expected_shape
			    img_batch = np.expand_dims(img, axis=0).reshape(batch_shape)
			    # Use input layer name if available to avoid warnings
			    input_names = getattr(model, 'input_names', None)
			    if input_names and isinstance(input_names, (list, tuple)) and len(input_names) == 1:
				pred = model.predict({input_names[0]: img_batch}, verbose=0)[0]
			    else:
				pred = model.predict(img_batch, verbose=0)[0]
			    elapsed = time.time() - start
			    total_time += elapsed
			    pred_class = np.argmax(pred)
			    true_class = np.argmax(label)
			    if pred_class == true_class:
				correct += 1
			    brier_scores.append(brier_score(pred, true_class))
			    entropies.append(softmax_entropy(pred))
			    margins.append(margin_score(pred))
			    # Ensure MC dropout gets a batch input
			    mc_mean, mc_var, _, _ = mc_dropout_predict(model, img_batch, T=10)
			    mc_vars.append(np.mean(mc_var))
		avg_time = total_time / len(images)
		accuracy = correct / len(images)
		result_row = {
			'model': model_name,
			'total_time': total_time,
			'avg_time': avg_time,
			'accuracy': accuracy,
			'brier_score': np.mean(brier_scores),
			'softmax_entropy': np.mean(entropies),
			'margin': np.mean(margins),
			'mc_dropout_var': np.mean(mc_vars)
		}
		results.append(result_row)
		print(f"Model: {model_name} | Total time: {total_time:.2f}s | Avg: {avg_time:.4f}s | Acc: {accuracy:.3f}")

		# Save after each model
		with open(RESULTS_CSV, 'a', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow(result_row)

	print(f"\nResults saved to {RESULTS_CSV}")

if __name__ == "__main__":
	main()
