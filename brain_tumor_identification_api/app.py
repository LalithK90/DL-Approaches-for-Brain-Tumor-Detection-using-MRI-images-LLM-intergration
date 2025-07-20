from flask import Flask, request, render_template, jsonify, session
from typing import Tuple, Dict, Any
import ollama
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
# This is from your old code, but we'll use a custom one
from tf_explain.core.grad_cam import GradCAM
import saliency.core as saliency
# This is from your old code, but we'll use a custom one
from saliency.core.xrai import XRAI
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import keras
import secrets

# Import the necessary components for the new GradCAM (likely from 'tf_keras_vis')
# Assuming you have tf-keras-vis installed: pip install tf-keras-vis
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

tf.data.experimental.enable_debug_mode()
tf.get_logger().setLevel('ERROR')

# Enable eager execution explicitly
tf.config.run_functions_eagerly(True)

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('model/propose_balance.h5')
    print("Model loaded successfully.")
except Exception as e:
    raise ValueError(f"Failed to load the model: {str(e)}")

IMG_SIZE = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

# --- New Functions from your request ---


def read_image(image_size, image_path):
    """
    Reads an image from a given path, applies CLAHE, bilateral filter,
    color mapping, and resizing.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_AREA)
    return img


def z_score_per_image(img_array):
    """
    Applies Z-score normalization per image.
    """
    mean = np.mean(img_array, axis=(1, 2, 3), keepdims=True)
    std = np.std(img_array, axis=(1, 2, 3), keepdims=True)
    return (img_array - mean) / (std + 1e-8)


def get_last_convolutional_layer(model):
    """
    Finds the name of the last convolutional layer in the model.
    """
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    return last_conv_layer_name


def generate_gradcam(img_array, model):
    """
    Generates a Grad-CAM heatmap for the given image and model.
    """
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
    # Ensure base_img is 3-channel for blending if it's grayscale
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    # Ensure base_img is uint8 and in range [0, 255] if it was float
    if base_img.dtype == np.float32 or base_img.dtype == np.float64:
        base_img = (base_img * 255).astype(np.uint8)

    superimposed_img = heatmap * 0.4 + base_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Return cam for Guided GradCAM (if you plan to implement it later)
    return superimposed_img, cam


def generate_saliency_map(img_array, model):
    """
    Generates a basic saliency map using gradients.
    """
    img_tensor = tf.convert_to_tensor(
        img_array, dtype=tf.float32)  # Ensure float32
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    saliency = (saliency - np.min(saliency)) / \
        (np.max(saliency) - np.min(saliency) + 1e-8)

    saliency = np.uint8(255 * saliency)
    saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    return saliency


def generate_lime(img_array, model):
    """
    Generates a LIME explanation for the given image and model.
    """
    # LIME explainer expects unnormalized image [0-255] and its own predict_fn
    # The input to model.predict should match the training data's preprocessing (e.g., [0,1] or normalized)
    # Here, we'll preprocess inside predict_fn to match our model's expectation
    def lime_predict_fn(images):
        # images from LIME are typically 0-1, so we convert them to match model's expected input
        processed_images = []
        for img in images:
            # Resize and apply z-score normalization as per our preprocess_image pipeline
            resized_img = cv2.resize(
                img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            # Ensure 3 channels if it's grayscale
            if len(resized_img.shape) == 2:
                resized_img = np.stack((resized_img,) * 3, axis=-1)
            # Apply Z-score if model expects it
            # The model expects batched input, so we add batch dimension here temporarily
            temp_array = np.expand_dims(resized_img, axis=0)
            normalized_array = z_score_per_image(
                temp_array.astype(np.float32))  # Ensure float32
            # Remove batch dim after normalization
            processed_images.append(normalized_array[0])

        return model.predict(np.array(processed_images))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        # LIME typically expects uint8 [0-255]
        (img_array[0] * 255).astype(np.uint8),
        lime_predict_fn,
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

    # Convert temp (LIME's image output, typically 0-1 float) back to uint8 for mark_boundaries if needed
    # mark_boundaries expects float [0,1] or uint8, it handles it.
    # LIME's temp is often centered around 0, so adjust
    lime_img = mark_boundaries(temp / 2 + 0.5, mask)
    # Convert to uint8 [0-255] for consistent output
    lime_img = np.uint8(lime_img * 255)

    return lime_img

# --- Modified Existing Functions ---


def preprocess_image_for_model(img_pil: Image.Image):
    """
    Preprocesses PIL image for model prediction using the new read_image and z_score_per_image functions.
    """
    # Save the PIL image to a temporary file path
    temp_img_path = "temp_upload_image.png"
    img_pil.save(temp_img_path)

    # Use the new read_image function
    img_processed = read_image(IMG_SIZE, temp_img_path)

    # Ensure it's 3 channels if it's grayscale from read_image (which applies colormap, so usually 3)
    if len(img_processed.shape) == 2:
        img_processed = np.stack((img_processed,) * 3, axis=-1)

    # Add batch dimension and apply Z-score normalization
    img_array = np.expand_dims(img_processed, axis=0)
    img_array = z_score_per_image(img_array.astype(
        np.float32))  # Ensure float32 for model input
    return img_array


def predict_tumor(image_pil: Image.Image):
    """
    Predicts tumor type using the preprocessed image.
    """
    processed_image_array = preprocess_image_for_model(image_pil)
    predictions = model(processed_image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    probabilities = predictions.numpy()[0]
    # Return img_array for explanations
    return CLASS_NAMES[predicted_class], probabilities, processed_image_array


def encode_image_to_base64(img_array):
    """
    Encodes a NumPy array image to a base64 string.
    Handles different dtypes and scales to 0-255 for encoding.
    """
    try:
        # Convert to 0-255 range and uint8 if not already
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            # If the image is a float and assumed to be [0,1], scale it.
            # If it's already a processed image (like GradCAM output), it might be 0-255.
            # We need to be careful here. Assuming explanation outputs are already in [0,1] or [0,255].
            # For simplicity, let's clip and scale to 0-255 if it's float.
            img_to_encode = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        elif img_array.dtype == np.uint8:
            img_to_encode = img_array
        else:
            raise ValueError(
                f"Unsupported image dtype: {img_array.dtype}. Must be float or uint8.")

        # Ensure 3 channels for consistent PNG saving, if it's grayscale
        if len(img_to_encode.shape) == 2:
            img_to_encode = cv2.cvtColor(img_to_encode, cv2.COLOR_GRAY2BGR)

        pil_img = Image.fromarray(img_to_encode)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        base64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_encoded

    except Exception as e:
        raise RuntimeError(f"Error encoding image to base64: {str(e)}")

# Remove XRAI function if you're not using it anymore, or keep it if needed for future
# def compute_xrai(image, prediction_class):
#     # ... (your existing XRAI code) ...
#     pass

# --- Flask Routes ---


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_pil = Image.open(file.stream).convert(
            'RGB')  # Ensure RGB for consistency

        # Pass the PIL image and get the preprocessed array back for explanations
        prediction, probabilities, processed_image_array = predict_tumor(
            image_pil)

        labeled_probabilities = {
            CLASS_NAMES[i]: str(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        # Generate explanations using the new functions
        grad_cam_img, _ = generate_gradcam(
            processed_image_array, model)  # _ for 'cam' value if not used
        lime_img = generate_lime(processed_image_array, model)
        saliency_map_img = generate_saliency_map(processed_image_array, model)

        response_data = {
            'prediction': prediction,
            'probabilities': labeled_probabilities,
            'grad_cam': encode_image_to_base64(grad_cam_img),
            'lime': encode_image_to_base64(lime_img),
            # New saliency map
            'saliency_map': encode_image_to_base64(saliency_map_img)
        }
        return jsonify(response_data)

    return jsonify({'error': 'Invalid file format'}), 400


app.secret_key = secrets.token_hex(32)


@app.route('/chat', methods=['POST'])
def chat():
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    data = request.get_json()
    medical_history = data.get('medical_history', '')
    pt_description = data.get('pt_description', '')
    symptoms = data.get('symptoms', '')
    prediction = data.get('prediction', '')
    user_message = data.get('message', '').strip()
    base64_image = data.get("file", "")

    if not user_message:
        return jsonify({"status": "error", "message": "No message provided"}), 400

    if not is_medical_related(user_message):
        # This line adds a note if not medical related, but still processes.
        # Consider if you want to completely block non-medical questions.
        user_message += " (Note: Please keep questions medically relevant)"

    image_path = None
    image_description = None
    if base64_image:
        try:
            image_path = save_base64_image(base64_image)
            if not image_path:
                return jsonify({"status": "error", "message": "Image decoding failed"}), 500

            image_response = ollama.chat(
                model='llama3.2-vision:latest',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this medical image in detail',
                    'images': [image_path]
                }]
            )
            image_description = image_response.message['content']
        except Exception as e:
            return jsonify({"status": "error", "message": f"Image analysis failed: {str(e)}"}), 500

    conversation_history = session['conversation_history']
    is_initial_request = len(conversation_history) == 0

    if is_initial_request:
        system_prompt = f"""
            You are RadioAI, a medical imaging specialist, and a medical expert in oncology. Analyze this case:
            Patient Overview:
            - Condition: {pt_description}
            - History: {medical_history}
            - Symptoms: {symptoms}
            Imaging Findings:
            - Preliminary Analysis: {image_description if image_description else "No image provided"}
            - AI Prediction: {prediction}
            - Always before taking AI prediction, check whether the image preliminary analysis matches the AI prediction or not. If not, provide the correct prediction.
            Image look like according to the AI prediction:
            - Describe the key features in the image
            - Explain the clinical implications
            Make sure to give responses according to the patient's medical history and symptoms, and finally, recommend consulting a doctor who specializes in the given patient condition, history, and symptoms.
            Instructions:
            1. Describe key features in the image
            2. Explain clinical implications
            3. Suggest next diagnostic steps
            4. If recommending treatment:
            - Use BNF-compliant medication names
            - Include standard dosages
            - Note contraindications
            5. How to {prediction} shown in the brain MRI ?
                - How to see in the brain MRI Axial View?
                - How to see in the brain MRI Coronal View?
                - How to see in the brain MRI Sagittal View?
            Explain the above views in detail, and is there way to show webite links for the above views?
            If the current query is not related to the medical field, ensure the query is related to the medical field.
            Provide a clear, actionable plan for the next steps in diagnosis and treatment.
        """
        messages = [{
            'role': 'user',
            'content': system_prompt,
            'images': [image_path] if base64_image else []
        }]
    else:
        messages = conversation_history.copy()
        messages.append({'role': 'user', 'content': user_message + """If the current query is not related to the medical field, ensure the query is related to the medical field.
            Provide a clear, actionable plan for the next steps in diagnosis and treatment, based on the patient's medical history and symptoms in the history."""})

    try:
        stream = ollama.chat(
            model='deepseek-r1:14b',
            messages=messages,
            stream=True,
            options={
                'temperature': 0.3,
                'num_ctx': 4096,
                'top_p': 0.9,
                'repeat_penalty': 1.1
            }
        )

        response_text = ""
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            print(content, end='', flush=True)
            response_text += content

        conversation_history.extend([
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': response_text}
        ])
        session['conversation_history'] = conversation_history

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def check_message(message):
    if is_medical_related(message):
        return True
    else:
        return "This is not a medical-related question, so please make sure that your question is medical-related. Thank you!"


def is_medical_related(message):
    medical_keywords = [
        # General Medical Terms
        "anatomy", "biology", "diagnosis", "disease", "healthcare", "medicine",
        "pathology", "physiology", "prescription", "prognosis", "symptom", "treatment",

        # Body Systems and Organs
        "brain", "heart", "lungs", "liver", "kidney", "stomach", "intestine", "pancreas",
        "thyroid", "nervous system", "cardiovascular", "respiratory", "digestive",
        "immune system", "musculoskeletal", "endocrine",

        # Medical Specialties
        "oncology", "cardiology", "neurology", "dermatology", "orthopedics", "pediatrics",
        "radiology", "surgery", "gynecology", "urology", "ophthalmology", "psychiatry",
        "pulmonology", "gastroenterology", "rheumatology", "nephrology",

        # Diseases and Conditions
        "cancer", "diabetes", "hypertension", "asthma", "arthritis", "infection",
        "inflammation", "tumor", "stroke", "heart attack", "pneumonia", "tuberculosis",
        "migraine", "epilepsy", "depression", "anxiety", "obesity", "anemia",
        "fibromyalgia", "osteoporosis", "alzheimer's", "parkinson's", "multiple sclerosis",
        "crohn's disease", "ulcerative colitis",

        # Symptoms
        "pain", "fever", "headache", "nausea", "vomiting", "dizziness", "fatigue", "cough",
        "shortness of breath", "chest pain", "abdominal pain", "swelling", "rash", "itching",
        "numbness", "weakness", "blurred vision", "insomnia", "diarrhea", "constipation",

        # Diagnostic Procedures
        "mri", "ct scan", "x-ray", "ultrasound", "biopsy", "blood test", "urine test",
        "ecg", "eeg", "endoscopy", "colonoscopy", "mammogram", "pet scan", "angiography",
        "lumbar puncture",

        # Treatments and Interventions
        "chemotherapy", "radiation therapy", "surgery", "medication", "antibiotics",
        "antiviral", "vaccine", "immunotherapy", "physical therapy", "occupational therapy",
        "dialysis", "transplantation", "hormone therapy", "laser treatment", "acupuncture",
        "chiropractic",

        # Medications and Drugs
        "aspirin", "ibuprofen", "acetaminophen", "insulin", "metformin", "statins",
        "antibiotics", "antihistamines", "antidepressants", "antipsychotics", "opioids",
        "steroids", "anticoagulants", "beta-blockers", "ace inhibitors",

        # Medical Imaging
        "radiology", "mri", "ct scan", "ultrasound", "x-ray", "fluoroscopy", "pet scan",
        "mammography", "tomography",

        # Patient Information
        "medical history", "family history", "allergies", "medications", "diagnosis",
        "prognosis", "treatment plan", "follow-up", "lab results", "vital signs", "bmi",

        # Healthcare Professionals
        "doctor", "physician", "surgeon", "nurse", "pharmacist", "radiologist", "oncologist",
        "therapist", "dentist", "optometrist", "psychiatrist", "pediatrician", "cardiologist",

        # Health and Wellness
        "diet", "nutrition", "exercise", "fitness", "weight loss", "mental health",
        "stress management", "sleep hygiene", "smoking cessation", "alcohol consumption",
        "hydration",

        # Emergency and First Aid
        "cpr", "first aid", "emergency", "trauma", "ambulance", "defibrillator", "aed",
        "poison control", "choking", "burns", "fractures",

        # Miscellaneous
        "clinical trial", "research study", "placebo", "side effects", "overdose",
        "addiction", "rehabilitation", "palliative care", "hospice", "telemedicine",
        "electronic health record"
    ]
    return any(keyword in message.lower() for keyword in medical_keywords)


def save_base64_image(base64_string, output_path="uploaded_image.jpg"):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)

        with open(output_path, "wb") as image_file:
            image_file.write(image_bytes)

        return output_path

    except Exception as e:
        print("Error decoding base64 image:", str(e))
        return None


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
