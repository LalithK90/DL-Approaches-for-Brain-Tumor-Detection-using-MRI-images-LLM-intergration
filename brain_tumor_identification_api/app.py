
from flask import Flask, request, render_template, jsonify,  session
from typing import Tuple, Dict, Any
import ollama
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tf_explain.core.grad_cam import GradCAM
from skimage.segmentation import quickshift
from skimage.segmentation import slic
# from tf_explain.core.smoothgrad import SmoothGrad
# from tf_explain.core.integrated_gradients import IntegratedGradients
# from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

import saliency.core as saliency
from saliency.core.xrai import XRAI
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import keras

from io import BytesIO

from PIL import Image
import numpy as np
import base64
from io import BytesIO
import secrets

tf.data.experimental.enable_debug_mode()
tf.get_logger().setLevel('ERROR')

# Enable eager execution explicitly
tf.config.run_functions_eagerly(True)

app = Flask(__name__)


try:
    # model = tf.keras.models.load_model('model/propose_224_model.h5')
    model = tf.keras.models.load_model('model/restnet50_inbalance_224-model.h5')
    # model = tf.keras.models.load_model('model/googleLeNet_inbalance_224-model.h5')
    # model = tf.keras.models.load_model('model/nin_inbalance_224-model.h5')
    # model = tf.keras.models.load_model('model/vgg16_inbalance_224-model.h5')
    print("Model loaded successfully.")
except Exception as e:
    raise ValueError(f"Failed to load the model: {str(e)}")


IMG_SIZE = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

def preprocess_image(img):
    img = np.array(img)
    img = cv2.bilateralFilter(img, d=2, sigmaColor=50, sigmaSpace=50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def get_last_convolutional_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def predict_tumor(image):
    processed_image = preprocess_image(image)
    predictions = model(processed_image) 
    predicted_class = np.argmax(predictions, axis=1)[0]
    probabilities = predictions.numpy()[0] 
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
    
    heatmap = grid.astype(np.float32)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)  
    
    return heatmap

def lime_explanation(image):
    processed_image = np.array(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)) / 255.0
    def custom_segmentation_slic(image):
        return slic(image, n_segments=100, compactness=10, sigma=1)
    
    # explainer = lime_image.LimeImageExplainer(feature_selection='auto', segmentation_fn=custom_segmentation_slic)

    def custom_segmentation(image):
        return quickshift(image, kernel_size=4, max_dist=2, ratio=0.5)
    
    # explainer = lime_image.LimeImageExplainer(feature_selection='auto', segmentation_fn=custom_segmentation)

    explainer = lime_image.LimeImageExplainer(feature_selection='auto')

    def predict_fn(x):
        return model.predict(x.reshape(-1, IMG_SIZE, IMG_SIZE, 3), batch_size=32)

    explanation = explainer.explain_instance(
        processed_image,
        predict_fn,
        top_labels=1,  
        hide_color=0,
        num_samples=100,  
        batch_size=32
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    return mark_boundaries(temp / 2 + 0.5, mask)



def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE)) 
    image = np.array(image)
    if len(image.shape) == 2:  
        image = np.stack((image,) * 3, axis=-1)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


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
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            raise ValueError("Image must be a NumPy array with dtype float or uint8.")

    
        pil_img = Image.fromarray(img)

       
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)  
       
        base64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_encoded

    except Exception as e:
        raise RuntimeError(f"Error encoding image to base64: {str(e)}")

def compute_xrai(image, prediction_class):

    try:
        processed_image = preprocess_image(image)
 
        xrai_object = saliency.XRAI()
 
        input_image = processed_image[0]
        if len(input_image.shape) == 2:
            input_image = np.stack((input_image,) * 3, axis=-1)
            
        xrai_attributions = xrai_object.GetMask(
            input_image,
            call_model_function,
            {'class_index': prediction_class}
        )
        
      
        if len(xrai_attributions.shape) == 2:
            xrai_attributions = np.expand_dims(xrai_attributions, axis=-1)
            xrai_attributions = np.repeat(xrai_attributions, 3, axis=-1)
        
     
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
            img = np.nan_to_num(img, nan=0.0)
            img = np.clip(img, 0, 1)
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
        user_message += " (Note: Please keep questions medically relevant)"

    image_description = None
    if base64_image:
        try:
            image_path = save_base64_image(base64_image)
            if not image_path:
                return jsonify({"status": "error", "message": "Image decoding failed"}), 500

            image_response = ollama.chat(
                model='llama3.2-vision:latest',
                # model='deepseek-r1:14b ',
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
        messages.append({'role': 'user', 'content': user_message +"""If the current query is not related to the medical field, ensure the query is related to the medical field.
            Provide a clear, actionable plan for the next steps in diagnosis and treatment, based on the patient's medical history and symptoms in the history."""})

    try:

        stream = ollama.chat(
            model='deepseek-r1:14b',  # Exact match from your installed models
            messages=messages,
            stream=True,
            options={
                # Best balance for this model (0.1-0.5)
                'temperature': 0.3,
                'num_ctx': 4096,          # Use full context window
                'top_p': 0.9,             # Controls response diversity
                'repeat_penalty': 1.1     # Reduce repetition
            }
        )


        response_text = ""
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            print(content, end='', flush=True)  # Proper streaming display
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