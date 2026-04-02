import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from brain_tumor_identification_api.src.models.models import (  # noqa: E402
	IMAGE_SIZE,
	LABELS,
	MODELS,
)
from brain_tumor_identification_api.src.utils.utils import (  # noqa: E402
	generate_gradcam,
	generate_lime,
	generate_saliency_map,
	read_image,
	softmax_entropy,
	z_score_per_image,
)


TESTING_FOLDER = os.path.join(PROJECT_ROOT, "xai_interpretation", "test_image")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "xai_interpretation", "generated")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_LABEL_INDEX = {
	"glioma": 0,
	"meningioma": 1,
	"notumor": 2,
	"pituitary": 3,
}


@dataclass(frozen=True)
class LoadedModel:
	model_name: str
	model: tf.keras.Model


def _resolve_model_path(model_path: str) -> str:
	return os.path.join(PROJECT_ROOT, "brain_tumor_identification_api", model_path)


def _pick_one_image_per_class(class_name: str) -> str:
	class_folder = os.path.join(TESTING_FOLDER, class_name)
	if not os.path.isdir(class_folder):
		raise FileNotFoundError(f"Class folder not found: {class_folder}")

	image_files = sorted(
		f
		for f in os.listdir(class_folder)
		if f.lower().endswith((".png", ".jpg", ".jpeg"))
	)
	if not image_files:
		raise FileNotFoundError(f"No images found in: {class_folder}")

	return os.path.join(class_folder, image_files[0])


def _normalize_for_plot(image: np.ndarray) -> np.ndarray:
	arr = np.asarray(image)
	if arr.ndim == 2:
		arr = np.stack([arr, arr, arr], axis=-1)

	if arr.dtype != np.uint8:
		arr = arr.astype(np.float32)
		arr_min, arr_max = float(np.min(arr)), float(np.max(arr))
		if arr_max > arr_min:
			arr = (arr - arr_min) / (arr_max - arr_min)
		arr = np.uint8(np.clip(arr * 255.0, 0, 255))

	# Most arrays in this project come from OpenCV and are BGR.
	return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _predict_with_model(model: tf.keras.Model, img_batch: np.ndarray) -> np.ndarray:
	# Some stored models expect named input dictionaries instead of bare arrays.
	input_name = None
	try:
		input_name = model.inputs[0].name.split(":")[0]
	except (AttributeError, IndexError):
		input_name = None

	if input_name:
		try:
			return model.predict({input_name: img_batch}, verbose=0)
		except Exception:
			pass

	return model.predict(img_batch, verbose=0)


def _safe_generate_visual(
	method_name: str,
	generator: Callable[[np.ndarray, tf.keras.Model], np.ndarray | tuple[np.ndarray, np.ndarray]],
	img_batch: np.ndarray,
	model: tf.keras.Model,
) -> np.ndarray:
	if method_name == "Grad-CAM":
		gradcam_img, _ = generator(img_batch, model)
		return _normalize_for_plot(gradcam_img)

	vis = generator(img_batch, model)
	return _normalize_for_plot(vis)


def _load_models() -> list[LoadedModel]:
	loaded_models: list[LoadedModel] = []
	for model_name, model_path in MODELS.items():
		resolved_path = _resolve_model_path(model_path)
		if not os.path.isfile(resolved_path):
			raise FileNotFoundError(f"Model file not found: {resolved_path}")

		model = tf.keras.models.load_model(resolved_path, compile=False)
		loaded_models.append(LoadedModel(model_name=model_name, model=model))
	return loaded_models


def _plot_method_grid(
	class_name: str,
	method_name: str,
	original_img_rgb: np.ndarray,
	model_visuals: list[tuple[str, np.ndarray]],
	output_dir: str,
) -> str:
	figure_title = f"{method_name} Interpretation for {class_name.capitalize()}"
	columns = 4
	rows = 4
	fig, axes = plt.subplots(rows, columns, figsize=(20, 20))
	axes_flat = axes.flatten()

	axes_flat[0].imshow(original_img_rgb)
	axes_flat[0].set_title("Given Image", fontsize=12)
	axes_flat[0].axis("off")


	# Place propose model output always next to the base image (index 1)
	propose_idx = None
	for i, (model_name, _) in enumerate(model_visuals):
		if model_name.startswith("propose"):
			propose_idx = i
			break
	visuals_ordered = model_visuals.copy()
	if propose_idx is not None:
		# Move propose model to first position
		propose_visual = visuals_ordered.pop(propose_idx)
		visuals_ordered.insert(0, propose_visual)

	for idx, (model_name, model_img) in enumerate(visuals_ordered, start=1):
		if idx >= len(axes_flat):
			break
		axes_flat[idx].imshow(model_img)
		axes_flat[idx].set_title(model_name, fontsize=10)
		axes_flat[idx].axis("off")

	used = min(len(visuals_ordered) + 1, len(axes_flat))
	for idx in range(used, len(axes_flat)):
		axes_flat[idx].axis("off")

	fig.suptitle(figure_title, fontsize=18, fontweight="bold")
	fig.tight_layout(rect=(0, 0, 1, 0.97))

	filename = f"{method_name.lower().replace('-', '_').replace(' ', '_')}_{class_name}.png"
	output_path = os.path.join(output_dir, filename)
	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	return output_path


def _save_comparison_csv(rows: list[dict[str, str | float]], output_dir: str) -> str:
	output_path = os.path.join(output_dir, "model_prediction_comparison.csv")
	fieldnames = [
		"class_name",
		"image_name",
		"model_name",
		"predicted_label",
		"predicted_confidence",
		"true_class_confidence",
		"entropy",
	]
	with open(output_path, "w", newline="") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	return output_path


def _save_summary_text(rows: list[dict[str, str | float]], output_dir: str) -> str:
	output_path = os.path.join(output_dir, "comparison_summary.txt")
	lines = []
	grouped: dict[str, list[dict[str, str | float]]] = {}
	for row in rows:
		grouped.setdefault(str(row["class_name"]), []).append(row)

	for class_name, class_rows in grouped.items():
		confidences = [float(r["true_class_confidence"]) for r in class_rows]
		agreements = sum(1 for r in class_rows if str(r["predicted_label"]).lower() == class_name)
		best = max(class_rows, key=lambda r: float(r["true_class_confidence"]))
		worst = min(class_rows, key=lambda r: float(r["true_class_confidence"]))

		lines.append(f"Class: {class_name}")
		lines.append(f"  Model agreement: {agreements}/{len(class_rows)}")
		lines.append(f"  Mean true-class confidence: {np.mean(confidences):.4f}")
		lines.append(
			"  Strongest model: "
			f"{best['model_name']} ({float(best['true_class_confidence']):.4f})"
		)
		lines.append(
			"  Weakest model: "
			f"{worst['model_name']} ({float(worst['true_class_confidence']):.4f})"
		)
		lines.append("")

	with open(output_path, "w") as text_file:
		text_file.write("\n".join(lines))
	return output_path


def main() -> None:
	methods: dict[str, Callable[[np.ndarray, tf.keras.Model], np.ndarray | tuple[np.ndarray, np.ndarray]]] = {
		"LIME": generate_lime,
		"Grad-CAM": generate_gradcam,
		"Saliency": generate_saliency_map,
	}

	run_output_dir = os.path.join(OUTPUT_ROOT, TIMESTAMP)
	os.makedirs(run_output_dir, exist_ok=True)

	loaded_models = _load_models()
	print(f"Loaded {len(loaded_models)} models.")

	comparison_rows: list[dict[str, str | float]] = []

	for class_name in CLASSES:
		image_path = _pick_one_image_per_class(class_name)
		image_name = os.path.basename(image_path)
		base_img = read_image(IMAGE_SIZE, image_path)
		img_batch = np.expand_dims(base_img, axis=0).astype(np.float32)
		img_batch = z_score_per_image(img_batch)
		original_img_rgb = _normalize_for_plot(base_img)

		model_visuals_by_method: dict[str, list[tuple[str, np.ndarray]]] = {
			method_name: [] for method_name in methods
		}

		print(f"Processing class: {class_name} | image: {image_name}")
		true_idx = CLASS_TO_LABEL_INDEX[class_name]

		for loaded in loaded_models:
			pred = _predict_with_model(loaded.model, img_batch)[0]
			pred_idx = int(np.argmax(pred))
			pred_label = LABELS[pred_idx]
			pred_conf = float(pred[pred_idx])
			true_conf = float(pred[true_idx])
			ent = float(softmax_entropy(pred))

			comparison_rows.append(
				{
					"class_name": class_name,
					"image_name": image_name,
					"model_name": loaded.model_name,
					"predicted_label": pred_label,
					"predicted_confidence": round(pred_conf, 6),
					"true_class_confidence": round(true_conf, 6),
					"entropy": round(ent, 6),
				}
			)

			for method_name, generator in methods.items():
				try:
					vis_rgb = _safe_generate_visual(method_name, generator, img_batch, loaded.model)
				except Exception as exc:
					print(f"  Failed {method_name} for {loaded.model_name}: {exc}")
					vis_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

				model_visuals_by_method[method_name].append((loaded.model_name, vis_rgb))

		for method_name in methods:
			figure_path = _plot_method_grid(
				class_name=class_name,
				method_name=method_name,
				original_img_rgb=original_img_rgb,
				model_visuals=model_visuals_by_method[method_name],
				output_dir=run_output_dir,
			)
			print(f"  Saved: {figure_path}")

	csv_path = _save_comparison_csv(comparison_rows, run_output_dir)
	txt_path = _save_summary_text(comparison_rows, run_output_dir)

	print("\nCompleted image-set generation.")
	print(f"Output folder: {run_output_dir}")
	print(f"Comparison CSV: {csv_path}")
	print(f"Summary TXT: {txt_path}")


if __name__ == "__main__":
	main()
