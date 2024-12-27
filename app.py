import os
import io
import re
import base64
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file
from lime import lime_image
from skimage.segmentation import mark_boundaries
from io import BytesIO
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore



IMAGE_SIZE=160

app = Flask(__name__)

# make new directory called model and change your model name 
try:
    model = load_model('model/model.h5')
except TypeError as e:
    print("Error Config:", e.args)


label_dict={0:'Glioma', 1:'Meningioma',2:'No tumor',3:'Pituitary'}

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	data = request.get_json(force=True)
	image = data['image']
	smooth_samples = data.get('smooth_samples', 50)
	smooth_noise = data.get('smooth_noise', 0.1)
	top_labels = data.get('top_labels', 5)
	hide_color = data.get('hide_color', 0)
	num_samples = data.get('num_samples', 1000)
	num_features = data.get('num_features', 5)
	min_weight = data.get('min_weight', 0.0)
    
	decoded = base64.b64decode(image)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	input_tensor = test_image

	prediction = model.predict(test_image)


	result=np.argmax(prediction,axis=1)[0]
	accuracy=str(np.max(prediction,axis=1)[0])

	label=label_dict[result]

		# Grad-CAM
	predictions = model.predict(input_tensor)
	predicted_class = np.argmax(predictions[0])
	gradcam = compute_gradcam(model, input_tensor, predicted_class)
	# Handle invalid values and normalize gradcam to [0, 1]
	gradcam = np.nan_to_num(gradcam)  # Replace NaN and Inf with 0
	gradcam_min = gradcam.min()
	gradcam_max = gradcam.max()
	if gradcam_max - gradcam_min > 0:
		gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min)  # Normalize
	else:
		gradcam = np.zeros_like(gradcam)  # If gradcam is constant, set all values to 0
	# Convert to uint8 for color mapping
	overlay = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
	overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

	def predict_proba(test_image):
		return model.predict(test_image)

    # LIME
	explainer = lime_image.LimeImageExplainer()
	explanation = explainer.explain_instance(
		image=np.squeeze(input_tensor),
		classifier_fn=predict_proba,
		top_labels=top_labels,
		hide_color=hide_color,
        num_samples=num_samples
	)
	lime_img, mask = explanation.get_image_and_mask(
		label=predicted_class,
		positive_only=True,
		hide_rest=False,
		num_features=num_features,
		min_weight=min_weight
	)
	lime_result = mark_boundaries(lime_img / 255, mask)

	# Saliency Map
	saliency_map = generate_saliency_map(model, input_tensor, predicted_class, smooth_samples, smooth_noise)

	# Combine Grad-CAM and LIME
	gradcam_3d = np.repeat(gradcam[:, :, np.newaxis], 3, axis=2)
	combined_img = (0.5 * gradcam_3d + 0.5 * lime_img / 255).clip(0, 1)

	# Generate visualization
	# buf = create_visualization(overlay, lime_result, combined_img, saliency_map)
	# encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
 
	# print(f'Prediction : {prediction}, Result : {result}, Accuracy : {accuracy}')
	# response = {'prediction': {'result' : label, 'accuracy' : accuracy}}
 
	# Generate visualization
	gradcam_img = (gradcam_3d * 255).astype(np.uint8)
	lime_explanation = (lime_img * 255).astype(np.uint8)
	lime_explanation_cam = (lime_result * 255).astype(np.uint8)
	combined_gradcam_lime_exp = (combined_img * 255).astype(np.uint8)
	saliency_map_img = (saliency_map * 255).astype(np.uint8)
 
	# Encode images
	gradcam_img_encoded = base64.b64encode(cv2.imencode('.png', gradcam_img)[1]).decode('utf-8')
	lime_explanation_encoded = base64.b64encode(cv2.imencode('.png', lime_explanation)[1]).decode('utf-8')
	combined_gradcam_lime_exp_encoded = base64.b64encode(cv2.imencode('.png', combined_gradcam_lime_exp)[1]).decode('utf-8')
	saliency_map_encoded = base64.b64encode(cv2.imencode('.png', saliency_map_img)[1]).decode('utf-8')
	lime_explanation_cam_encoded = base64.b64encode(cv2.imencode('.png', lime_explanation_cam)[1]).decode('utf-8')

 
	response = {
        'prediction': {'result': label, 'accuracy': accuracy},
        'gradcam_img': gradcam_img_encoded,
        'lime_explanation': lime_explanation_encoded,
        'combined_gradcam_lime_exp': combined_gradcam_lime_exp_encoded,
        'saliency_map': saliency_map_encoded,
        'lime_explanation_cam': lime_explanation_cam_encoded
    }

	return jsonify(response)



# Get Last Convolutional Layer Name
def get_last_convolutional_layer(model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    return last_conv_layer_name

# image process 
def preprocess(img):
	img = np.array(img)
	img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
	# Apply a colormap (COLORMAP_BONE in this case)
	img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
	# Resize the image to the target size
	img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
	# Convert the image to a Keras-compatible array
	array = keras.utils.img_to_array(img)
	# Expand dimensions to match the input shape required by the model
	array = np.expand_dims(array, axis=0)
	return array


# Grad-CAM Function
def compute_gradcam(model, img_tensor, class_index):
    last_conv_layer_name = get_last_convolutional_layer(model)
    
    # Handle multiple outputs
    model_output = model.output[0] if isinstance(model.output, (list, tuple)) else model.output
    
    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output, 
            model_output
        ]
    )
    
    # Compute Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    gradcam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled_grads):
        gradcam += w * conv_outputs[:, :, i]

    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (IMAGE_SIZE,IMAGE_SIZE))
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()
    return gradcam

# Saliency Map Function# Generate Saliency Map
def generate_saliency_map(model, input_tensor, class_index,smooth_samples,smooth_noise):
    score = CategoricalScore([class_index])  # Target the predicted class
    saliency = Saliency(model, clone=False)
    saliency_map = saliency(score, input_tensor, smooth_samples=smooth_samples, smooth_noise=smooth_noise)
    return normalize(saliency_map[0])

def create_visualization(overlay, lime_result, combined_img, saliency_map):
    """Generate visualization plots and save them to a buffer."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Grad-CAM")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(lime_result)
    axes[0, 1].set_title("LIME Explanation")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(combined_img)
    axes[1, 0].set_title("Combined Grad-CAM & LIME")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(saliency_map, cmap="hot")
    axes[1, 1].set_title("Saliency Map")
    axes[1, 1].axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # Ensure the figure is closed after saving
    return buf

if __name__ == "__main__":
    app.run(debug=True)