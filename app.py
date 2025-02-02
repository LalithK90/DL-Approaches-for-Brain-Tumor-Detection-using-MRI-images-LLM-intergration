import io
import os
from flask import Flask, request, render_template, jsonify, send_file, session
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tf_explain.core.grad_cam import GradCAM
# from tf_explain.core.smoothgrad import SmoothGrad
# from tf_explain.core.integrated_gradients import IntegratedGradients
# from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

import saliency.core as saliency
from saliency.core.xrai import XRAI
from saliency.core.xrai import XRAI

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import keras

from io import BytesIO

from PIL import Image
import numpy as np
import base64
from io import BytesIO
import secrets
import requests

tf.data.experimental.enable_debug_mode()

# Enable eager execution explicitly
tf.config.run_functions_eagerly(True)

app = Flask(__name__)


try:
    # model = tf.keras.models.load_model('model/propose_224_model.h5')
    # model = tf.keras.models.load_model('model/restnet50_inbalance-model.h5')
    model = tf.keras.models.load_model('model/googleLeNet_inbalance-model.h5')
    # model = tf.keras.models.load_model('model/nin_inbalance-model.h5')
    # model = tf.keras.models.load_model('model/vgg16_inbalance-model.h5')
    print("Model loaded successfully.")
except Exception as e:
    raise ValueError(f"Failed to load the model: {str(e)}")


IMG_SIZE = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

def preprocess_image(img):
    img = np.array(img)
    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
	# Apply a colormap (COLORMAP_BONE in this case)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
	# Resize the image to the target size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
	# Convert the image to a Keras-compatible array
    array = keras.utils.img_to_array(img)
	# Expand dimensions to match the input shape required by the model
    array = np.expand_dims(array, axis=0)
    return array

def get_last_convolutional_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def predict_tumor(image):
    processed_image = preprocess_image(image)
    predictions = model(processed_image)  # Use model as a callable
    predicted_class = np.argmax(predictions, axis=1)[0]
    probabilities = predictions.numpy()[0]  # Convert EagerTensor to NumPy array
    return CLASS_NAMES[predicted_class], probabilities

def grad_cam_explanation(image):
    processed_image = preprocess_image(image)
    predictions = model(processed_image)
    predicted_class_index = np.argmax(predictions.numpy(), axis=1)[0]
    
    explainer = GradCAM()
    grid = explainer.explain(
        validation_data=(processed_image, None),
        model=model,
        class_index=predicted_class_index,
        layer_name=get_last_convolutional_layer(model)
    )
    
    # Normalize the heatmap safely
    heatmap = grid.astype(np.float32)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)  # Handle invalid heatmaps gracefully
    
    return heatmap

def lime_explanation(image):
    # Preprocess image once
    processed_image = np.array(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)) / 255.0

    # Create explainer with optimized settings
    explainer = lime_image.LimeImageExplainer(feature_selection='auto')

    # Define prediction function outside explain_instance for better performance
    def predict_fn(x):
        return model.predict(x.reshape(-1, IMG_SIZE, IMG_SIZE, 3), batch_size=32)

    # Get explanation with optimized parameters
    explanation = explainer.explain_instance(
        processed_image,
        predict_fn,
        top_labels=1,  # Reduced from 4 since we only use first label
        hide_color=0,
        num_samples=10,  # Reduced samples for better performance while maintaining quality
        batch_size=32
    )

    # Get visualization
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=4,
        hide_rest=False
    )

    return mark_boundaries(temp / 2 + 0.5, mask)

# Define preprocessing function
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize to match model's input size
    image = np.array(image)
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack((image,) * 3, axis=-1)
    image = image / 255.0   # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define call_model_function for saliency computation
def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args['class_index']
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(images)
        tape.watch(inputs)
        predictions = model(inputs)
        output_layer = predictions[:, target_class_idx]
    gradients = tape.gradient(output_layer, inputs)
    return {saliency.INPUT_OUTPUT_GRADIENTS: gradients}

def encode_image_to_base64(img):
    try:
        # Ensure the image is in the range [0, 255] and convert to uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            raise ValueError("Image must be a NumPy array with dtype float or uint8.")

        # Convert the NumPy array to a PIL Image
        pil_img = Image.fromarray(img)

        # Save the PIL Image to a BytesIO buffer in PNG format
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)  # Rewind the buffer to the beginning

        # Encode the buffer content to base64
        base64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_encoded

    except Exception as e:
        raise RuntimeError(f"Error encoding image to base64: {str(e)}")

def compute_xrai(image, prediction_class):

    try:
        processed_image = preprocess_image(image)
        
        # Step 4: Compute XRAI attributions
        xrai_object = saliency.XRAI()
        # Ensure input is 3D (height, width, channels)
        input_image = processed_image[0]
        if len(input_image.shape) == 2:
            input_image = np.stack((input_image,) * 3, axis=-1)
            
        xrai_attributions = xrai_object.GetMask(
            input_image,
            call_model_function,
            {'class_index': prediction_class}
        )
        
        # Reshape attributions to include channel dimension if necessary
        if len(xrai_attributions.shape) == 2:
            xrai_attributions = np.expand_dims(xrai_attributions, axis=-1)
            xrai_attributions = np.repeat(xrai_attributions, 3, axis=-1)
        
        # Step 5: Visualize the XRAI attributions
        if len(xrai_attributions.shape) == 2:
            xrai_attributions_3d = np.expand_dims(xrai_attributions, axis=-1)
            xrai_attributions_3d = np.repeat(xrai_attributions_3d, 3, axis=-1)
        else:
            xrai_attributions_3d = xrai_attributions
            
        grayscale_viz = saliency.VisualizeImageGrayscale(xrai_attributions_3d)
        overlay_viz = saliency.VisualizeImageDiverging(xrai_attributions_3d)
       
        
        return {
            'grayscale_viz': grayscale_viz,
            'overlay_viz': overlay_viz
        }
    
    except Exception as e:
        raise RuntimeError(f"Error computing XRAI: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file.stream)
        prediction, probabilities = predict_tumor(image)

        labeled_probabilities = {
            CLASS_NAMES[i]: str(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        grad_cam = grad_cam_explanation(image)
        lime_img = lime_explanation(image)
        # explanation_ig = generate_integrated_gradients(image)
        # explanation_occlusion = generate_occlusion_sensitivity(image)
        # explanation_scorecam = generate_scorecam(image)

        # Convert explanations to base64
        def to_base64(img):
            buffer = BytesIO()
            # Normalize image to [0,1] range and handle NaN values
            img = np.nan_to_num(img, nan=0.0)
            img = np.clip(img, 0, 1)
            # Convert to uint8 range [0,255]
            img_uint8 = (img * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Compute XRAI attributions
        xrai_result = compute_xrai(image, np.argmax(probabilities))

        response_data = {
            'prediction': prediction,
            'probabilities': labeled_probabilities,
            'grad_cam': to_base64(grad_cam),
            'lime': to_base64(np.array(lime_img)),
            'grayscale_viz': to_base64(xrai_result['grayscale_viz']),
            'overlay_viz': to_base64(xrai_result['overlay_viz'])
        }
        return jsonify(response_data)

    return jsonify({'error': 'Invalid file format'}), 400

app.secret_key = secrets.token_hex(32)

# Default Ollama API URL
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")

@app.route('/chat', methods=['POST'])
def chat():
    # Initialize session for conversation history if not already present
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Extract form data
    medical_history = request.form.get('medical_history', '')
    pt_description = request.form.get('pt_description', '')
    symptoms = request.form.get('symptoms', '')
    prediction = request.form.get('prediction', '')
    user_message = request.form.get('message', '')

    # Handle file if it exists
    file_data = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_data = base64.b64encode(file.read()).decode('utf-8')

    # Debugging logs
    print(f"Medical History: {medical_history}")
    print(f"Patient Description: {pt_description}")
    print(f"Symptoms: {symptoms}")
    print(f"Prediction: {prediction}")
    print(f"User Message: {user_message}")
    print(f"File: {file_data if file_data else 'No file'}")

    # Build the payload for Ollama API
    conversation_history = session['conversation_history']
    conversation_history.append({"role": "user", "content": user_message})

    payload = {
        "model": "llama3.2-vision:latest",  # Replace with your preferred model
        "prompt": f"""
            You are named RadioAI.
            You are a medical expert in oncology. Based on the MRI image analysis, the patient has been diagnosed with cancer.
            The AI has identified a brain tumor. Please provide step-by-step guidance for the next steps:
            
            Patient Information (if available):
            - Patient Details: {pt_description}
            - Medical History: {medical_history}
            - Symptoms: {symptoms}
            
            MRI Analysis Results:
            - Tumor Type: {prediction}
            
            Uploaded Image: {file_data} if there is image, describe this image
            
            Previous Conversation History:
            {conversation_history}
            
            Current User Message: {user_message} if there is no user message that means this is the first message so make sure to give propoper descriptinve alalysis about the case if you need more details please ask the user
            
            Provide a clear, actionable plan for the next steps in diagnosis and treatment.
        """
    }

    # Prepare the Ollama API request
    ollama_request = {
        "model": "llama3.2-vision:latest",
        "prompt": payload,
        "stream": False
    }

    try:
        # Send the request to Ollama API
        ollama_response = requests.post(OLLAMA_API_URL, json=ollama_request)
        ollama_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        ollama_data = ollama_response.json()
        response_text = ollama_data.get("response", "Sorry, I couldn't generate a response.")
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to communicate with Ollama API: {str(e)}'}), 500

    # Append the AI's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_text})
    session['conversation_history'] = conversation_history

    # Return the AI's response
    return jsonify({'response': response_text})

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

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)