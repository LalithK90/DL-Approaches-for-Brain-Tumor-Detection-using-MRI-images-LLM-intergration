from flask import Blueprint, render_template, request, jsonify, url_for, current_app, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import json
from src.models.models import load_model, LABELS, IMAGE_SIZE
from src.utils.utils import (
    allowed_file, read_image, z_score_per_image, generate_gradcam,
    generate_saliency_map, generate_lime, analyze_gradcam_heatmap,
    get_last_convolutional_layer, generate_gradcam_tf_explain,
    get_top3_probs, softmax_entropy, margin_score, mc_dropout_predict,
    brier_score, dice_iou
)
from src.LLM.ollama_client import (get_text_reasoning,
                                   get_image_description_llama3_vision,
                                   get_medical_report_from_text_medgemma,
                                   get_medical_report_from_image_medgemma)


main_bp = Blueprint('main', __name__)


@main_bp.route('/main')
def index():
    return render_template('index.html')


def _process_and_predict(model, img_array):
    """Helper to run model prediction and get initial results."""
    prediction = model.predict(img_array)
    pred = prediction[0]
    class_idx = np.argmax(pred)
    predicted_class = LABELS[class_idx]
    confidence_val = float(np.max(pred))
    return pred, class_idx, predicted_class, confidence_val


def _generate_visualizations(model, img_array, class_idx, filename):
    """Helper to generate all XAI visualizations."""
    try:
        gradcam_img, cam = generate_gradcam(img_array, model)
        saliency_img = generate_saliency_map(img_array, model)
        lime_img = generate_lime(img_array, model)

        layer_name = get_last_convolutional_layer(model)
        tfexplain_filename = f"tfexplain_gradcam_{filename}"
        gradcam_analysis_path = os.path.join(
            current_app.config['VISUALIZATION_FOLDER'], tfexplain_filename)
        generate_gradcam_tf_explain(
            model, img_array, class_idx, layer_name, gradcam_analysis_path)

        return {
            "gradcam_img": gradcam_img,
            "saliency_img": saliency_img,
            "lime_img": lime_img,
            "cam": cam,
            "tfexplain_filename": tfexplain_filename
        }
    except ValueError as e:
        # Propagate the error to be handled by the main route
        raise e


def _calculate_metrics(pred, cam, lime_img, model, img_array, labels, filename):
    """Helper to compute all quantitative metrics."""
    # Base metrics
    top3 = get_top3_probs(pred, labels)
    entropy_val = softmax_entropy(pred)
    margin = margin_score(pred)
    brier = brier_score(pred, np.argmax(pred))

    # Construct the full path for the analysis image
    analysis_filename = f"com_analysis_{filename}"
    analysis_filepath = os.path.join(
        current_app.config['VISUALIZATION_FOLDER'], analysis_filename)
    # Grad-CAM analysis metrics
                
    center_distance, activation_ratio = analyze_gradcam_heatmap(
        cam, analysis_filepath)

    # MC Dropout metrics
    mc_mean, mc_var, mc_ci_low, mc_ci_high = mc_dropout_predict(
        model, img_array, T=30)
    mc_confidence_interval = [
        float(mc_ci_low[np.argmax(mc_mean)]), float(mc_ci_high[np.argmax(mc_mean)])]
    mc_variance = float(mc_var[np.argmax(mc_mean)])

    # Agreement metrics
    gradcam_mask = (cam > 127).astype(np.uint8)
    lime_mask = (lime_img[..., 0] > 127).astype(np.uint8)
    dice_score, iou_score = dice_iou(gradcam_mask, lime_mask)

    return {
        'top3': top3,
        'entropy': get_metric_interpretation('entropy', entropy_val),
        'margin': get_metric_interpretation('margin', margin),
        'brier': get_metric_interpretation('brier', brier),
        'center_distance': center_distance,
        'activation_ratio': activation_ratio,
        'mc_confidence_interval': mc_confidence_interval,
        'mc_variance': get_metric_interpretation('mc_variance', mc_variance),
        'dice': get_metric_interpretation('dice', dice_score),
        'iou': get_metric_interpretation('iou', iou_score),
    }


def _save_visualizations(visuals, original_img, filename):
    """Helper to save generated images to disk."""
    vis_folder = current_app.config['VISUALIZATION_FOLDER']
    cv2.imwrite(os.path.join(vis_folder, f"original_{filename}"), original_img)
    cv2.imwrite(os.path.join(
        vis_folder, f"gradcam_{filename}"), visuals['gradcam_img'])
    cv2.imwrite(os.path.join(
        vis_folder, f"saliency_{filename}"), visuals['saliency_img'])
    cv2.imwrite(os.path.join(
        vis_folder, f"lime_{filename}"), visuals['lime_img'])


def _validate_file_upload(request):
    if 'file' not in request.files:
        return None, jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return None, jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return None, jsonify({'error': 'File type not allowed'}), 400

    return file, None, None

def _save_uploaded_file(file, filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

def _process_image(filepath, model_name):
    img = read_image(IMAGE_SIZE, filepath)
    img_array = z_score_per_image(np.array([img]))
    model = load_model(model_name)
    if model is None:
        return None, jsonify({'error': 'Model not found'}), 404
    return img, img_array, model

def _generate_response_data(filename, predicted_class, confidence_val, patient_info, visuals):
    response_data = {
        'original': url_for('static', filename=f'uploads/{filename}', _external=True),
        'gradcam': url_for('static', filename=f'visualizations/gradcam_{filename}', _external=True),
        'saliency': url_for('static', filename=f'visualizations/saliency_{filename}', _external=True),
        'lime': url_for('static', filename=f'visualizations/lime_{filename}', _external=True),
        'gradcam_analysis': url_for('static', filename=f'visualizations/com_analysis_{filename}', _external=True),
        'gradcam_heatmap': url_for('static', filename=f"visualizations/{visuals['tfexplain_filename']}", _external=True),
        'prediction': predicted_class,
        'confidence': get_metric_interpretation('confidence', confidence_val),
        'patient_info': patient_info,
        'xai_educational_notes': {
            'gradcam_explanation': "Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions in the image that most strongly influenced the model's prediction. Brighter areas indicate features that strongly support the diagnosis.",
            'saliency_explanation': "Saliency maps show which pixels in the image had the greatest influence on the classification decision. They help identify the specific image features the model focused on.",
            'lime_explanation': "Local Interpretable Model-agnostic Explanations (LIME) creates a simplified local model to explain which features contributed most to the prediction. It helps understand which image regions support or contradict the diagnosis.",
            'metrics_explanation': "Quantitative metrics provide objective measures of model confidence and uncertainty. They help assess the reliability of the AI diagnosis and identify cases that may require additional clinical correlation."
        }
    }
    return response_data

def _generate_final_report(filepath, predicted_class, patient_info, metrics, llama3_response):
    patient_info_string = json.dumps(patient_info, indent=2)
    metrics_string = json.dumps(metrics, indent=2)

    final_report_prompt_medgemma = f"""
        Synthesize into a comprehensive brain tumor diagnosis report:
        - Source 1: AI Image Analysis (Llama3.2-vision): {llama3_response}
        - Source 2: Patient Information: {patient_info_string}
        - Source 3: Quantitative Metrics: {metrics_string}
        - Source 4: Primary Diagnosis: {predicted_class}
        Sections:
        1. Clinical Presentation: Summarize demographics, symptoms, history.
        2. Imaging Findings: Describe lesions, anatomy, measurements, effects.
        3. Quantitative Assessment: Interpret metrics for certainty.
        4. Differential Diagnosis: Ranked list with supporting features.
        5. Pathophysiology: Disease mechanisms and characteristics.
        6. Clinical Implications: Symptoms, progression, complications.
        7. Management: Evidence-based treatments and outcomes.
        8. Educational Pearls: 3-5 key points on diagnostics and pitfalls.
        9. References: Relevant guidelines/studies.
        Use precise terminology with explanations. Aim for accurate diagnosis and education.
    """

    medgemma_text_response = get_medical_report_from_image_medgemma(
        final_report_prompt_medgemma, filepath)

    final_report_prompt_deepsek = f"""
        Create a final educational brain tumor diagnosis report:
        - Source 1: MedGemma Analysis: {medgemma_text_response}
        - Source 2: Llama3.2-vision Analysis: {llama3_response}
        - Source 3: Patient Info: {patient_info_string}
        - Source 4: Metrics: {metrics_string}
        - Source 5: Primary Diagnosis: {predicted_class}
        - Source 6: XAI Visualizations: Grad-CAM, Saliency Map, LIME - explain key regions.
        Sections:
        1. Executive Summary: 2-3 sentence overview.
        2. Clinical Presentation: Demographics, symptoms, history.
        3. Imaging Findings: Descriptions, anatomy, XAI highlights.
        4. Quantitative Assessment: Interpret metrics, discrepancies.
        5. Differential Diagnosis: Ranked list with features.
        6. Pathophysiology: Mechanisms, markers.
        7. Clinical Implications: Symptoms, prognosis.
        8. Management: Treatments, protocols.
        9. Educational Pearls: 3-5 points on diagnostics, pitfalls.
        10. References: Guidelines/studies.
        Guidelines: Precise terms with explanations; detailed images; headings/bullets; highlight keys; connect findings; explain XAI; aim for diagnosis and education.
    """

    resoning_final_report = get_text_reasoning(final_report_prompt_deepsek, filepath)
    return resoning_final_report

@main_bp.route('/predict', methods=['POST'])
def predict():
    # Validate file upload
    file, error_response, status_code = _validate_file_upload(request)
    if error_response:
        return error_response, status_code

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    filepath = _save_uploaded_file(file, filename)
    session['current_image'] = filename.replace(" ", "")
    current_image=filename.replace(" ", "")
    print(f"File saved at: {current_image}")
    # Process the image and load the model
    model_name = request.form.get('model_name', 'propose_balance')
    img, img_array, model = _process_image(filepath, model_name)
    if model is None:
        return jsonify({'error': 'Model not found'}), 404

    # Get model prediction
    pred, class_idx, predicted_class, confidence_val = _process_and_predict(model, img_array)

    # Generate and save XAI visualizations
    try:
        visuals = _generate_visualizations(model, img_array, class_idx, filename)
        _save_visualizations(visuals, img, filename)
    except ValueError as e:
        return jsonify({'error': str(e)})

    # Calculate metrics
    metrics = _calculate_metrics(pred, visuals['cam'], visuals['lime_img'], model, img_array, LABELS, filename)

    # Find matching patient data
    with open(current_app.config['PATIENT_DATA_PATH'], 'r') as f:
        patient_data = json.load(f)
    patient_info = find_patient_by_diagnosis(predicted_class, patient_data)
    del patient_info["diagnosis"]

    # Assemble the response data
    response_data = _generate_response_data(filename, predicted_class, confidence_val, patient_info, visuals)

    # Generate the final report
    image_analysis_prompt = """Analyze brain MRI as expert neuroradiologist:
1. Anatomical localization of abnormalities
2. Signal characteristics
3. Lesion size/shape
4. Mass effect/edema/shift
5. Enhancement patterns
6. Critical structure involvement
7. Secondary findings
8. Specific signs for diagnosis
Use precise terminology with explanations."""

    # llama3_response = get_image_description_llama3_vision(image_analysis_prompt, filepath)
    final_report = _generate_final_report(filepath, predicted_class, patient_info, metrics, "llama3_response")

    response_data.update(metrics)
    response_data['final_report'] = final_report

    return jsonify(response_data)


def get_metric_interpretation(metric_name, value):
    """Provides a detailed qualitative interpretation and level for a given metric value.
    Includes educational explanations to help practitioners understand the significance."""
    interpretation = "N/A"
    level = "neutral"  # 'good', 'warning', 'bad'
    explanation = ""  # Educational explanation of the metric

    if metric_name == 'confidence':
        if value >= 0.9:
            interpretation = "Very High"
            level = 'good'
            explanation = "The model is extremely confident in its diagnosis, suggesting strong characteristic features of this tumor type are present in the image."
        elif value >= 0.7:
            interpretation = "High"
            level = 'good'
            explanation = "The model shows strong confidence in its diagnosis, indicating clear presence of typical features for this tumor type."
        elif value >= 0.5:
            interpretation = "Moderate"
            level = 'warning'
            explanation = "The model shows moderate confidence, suggesting some typical features are present but possibly with atypical characteristics that create ambiguity."
        else:
            interpretation = "Low"
            level = 'bad'
            explanation = "The model has low confidence in its diagnosis, indicating this may be an atypical presentation or the image may contain features common to multiple tumor types."
    
    elif metric_name == 'entropy':
        max_entropy = np.log2(4)  # For 4 classes
        if value <= max_entropy * 0.25:
            interpretation = "Low Uncertainty"
            level = 'good'
            explanation = "The probability distribution across possible classes is concentrated, indicating the model is decisive in its classification with minimal uncertainty."
        elif value <= max_entropy * 0.6:
            interpretation = "Moderate Uncertainty"
            level = 'warning'
            explanation = "The model shows some distribution of probability across multiple classes, suggesting features common to different tumor types may be present."
        else:
            interpretation = "High Uncertainty"
            level = 'bad'
            explanation = "The model's probability is widely distributed across multiple classes, indicating significant diagnostic uncertainty. This may require additional imaging sequences or histopathological confirmation."
    
    elif metric_name == 'margin':
        if value >= 0.7:
            interpretation = "Decisive"
            level = 'good'
            explanation = "There is a large margin between the top prediction and alternatives, indicating the model strongly favors this diagnosis over others."
        elif value >= 0.3:
            interpretation = "Reasonable"
            level = 'warning'
            explanation = "The margin between the top prediction and alternatives is moderate, suggesting some distinctive features but also some overlapping characteristics with other tumor types."
        else:
            interpretation = "Indecisive"
            level = 'bad'
            explanation = "The small margin between top predictions indicates the model finds it difficult to distinguish between multiple possible diagnoses. Consider additional diagnostic methods."
    
    elif metric_name == 'dice' or metric_name == 'iou':
        if value >= 0.7:
            interpretation = "High Agreement"
            level = 'good'
            explanation = "Strong spatial agreement between different explainability methods, indicating consistent identification of relevant image regions."
        elif value >= 0.4:
            interpretation = "Moderate Agreement"
            level = 'warning'
            explanation = "Partial agreement between explainability methods, suggesting some consistency in identified regions but also some differences in feature importance."
        else:
            interpretation = "Low Agreement"
            level = 'bad'
            explanation = "Limited agreement between explainability methods, indicating uncertainty in which image features are most relevant for diagnosis."
    
    elif metric_name == 'mc_variance':
        if value <= 0.01:
            interpretation = "Very Stable"
            level = 'good'
            explanation = "Monte Carlo dropout shows very low variance, indicating high model stability and consistency in predictions across stochastic forward passes."
        elif value <= 0.05:
            interpretation = "Moderately Stable"
            level = 'warning'
            explanation = "Some variance in Monte Carlo dropout predictions, suggesting moderate model stability with some sensitivity to dropout perturbations."
        else:
            interpretation = "Unstable"
            level = 'bad'
            explanation = "High variance in Monte Carlo dropout predictions, indicating model instability and high sensitivity to the dropout perturbation, suggesting uncertainty in the diagnosis."
    
    elif metric_name == 'brier':
        if value <= 0.1:
            interpretation = "Excellent Calibration"
            level = 'good'
            explanation = "Low Brier score indicates excellent calibration between predicted probabilities and actual outcomes, suggesting reliable confidence estimates."
        elif value <= 0.25:
            interpretation = "Good Calibration"
            level = 'warning'
            explanation = "Moderate Brier score suggests reasonable but not perfect calibration between predicted probabilities and actual outcomes."
        else:
            interpretation = "Poor Calibration"
            level = 'bad'
            explanation = "High Brier score indicates poor calibration, suggesting the model's confidence may not reliably reflect actual diagnostic accuracy."

    return {"value": float(value), "interpretation": interpretation, "level": level, "explanation": explanation}


def find_patient_by_diagnosis(tumor_type, patient_data):
    diagnosis_map = {
        'Glioma': 'glioma',
        'Meningioma': 'meningioma',
        'Notumor': 'no_tumor',
        'Pituitary': 'pituitary'
    }
    target_diagnosis = diagnosis_map.get(tumor_type)
    if not target_diagnosis:
        return None

    for patient in patient_data.get('patients', []):
        if patient.get('diagnosis', {}).get('tumor_type') == target_diagnosis:
            return patient
    return None


@main_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided or invalid request format'}), 400

    message = data['message']
    image_name = session.get('current_image')
   # If no image name in session, try to get it from the request data
    if not image_name:
        print("No image found in session, checking request data")
        if 'image' in data:
            image_name = data['image']
        else:
            print("No image found in request data")
            return jsonify({'error': 'No image uploaded or session expired'}), 400

    prompt = f"""
                As expert neuroradiologist and neurosurgeon in brain tumors, answer the practitioner's question about the uploaded MRI scan. Provide detailed, educational response addressing the query directly, with insights on diagnosis and management.

                USER QUESTION: {message}
                IMAGE: {image_name}
                Output flexibly based on query; use headings/bullets as needed for clarity and education.
            """
    text_response = get_text_reasoning(
        prompt, image_name)
    if text_response:
        return jsonify({'response': text_response})
    else:
        return jsonify({'error': 'Failed to get a response from the AI model.'}), 500
