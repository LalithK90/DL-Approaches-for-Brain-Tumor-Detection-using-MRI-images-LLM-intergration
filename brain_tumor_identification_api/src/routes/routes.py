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


@main_bp.route('/')
@login_required
def index():
    return render_template('index.html')


def _process_and_predict(model, img_array):
    """Helper to run model prediction and get initial results."""
    prediction = model.predict([img_array])
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
            model, img_array[0], class_idx, layer_name, gradcam_analysis_path)

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


@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        #  data base save
        # need to manage session to store the current image then able to take all details related to the image
        session['current_image'] = filename.replace(" ", "")

        model_name = request.form.get('model_name', 'propose_balance')
        model = load_model(model_name)
        if model is None:
            return jsonify({'error': 'Model not found'}), 404

        img = read_image(IMAGE_SIZE, filepath)
        img_array = z_score_per_image(np.array([img]))

        # 1. Get model prediction
        pred, class_idx, predicted_class, confidence_val = _process_and_predict(
            model, img_array)

        # 2. Generate XAI visualizations
        try:
            visuals = _generate_visualizations(
                model, img_array, class_idx, filename)
        except ValueError as e:
            return jsonify({'error': str(e)})

        # 3. Save visualizations to disk
        _save_visualizations(visuals, img, filename)

        # 4. Calculate all quantitative metrics
        metrics = _calculate_metrics(
            pred, visuals['cam'], visuals['lime_img'], model, img_array, LABELS, filename)

        # 5. Find matching patient data
        with open(current_app.config['PATIENT_DATA_PATH'], 'r') as f:
            patient_data = json.load(f)
        patient_info = find_patient_by_diagnosis(predicted_class, patient_data)
        del patient_info["diagnosis"]

        # 6. Assemble and return the final JSON response
        response_data = {
            'original': url_for('static', filename=f'/uploads/{filename}', _external=True),
            'gradcam': url_for('static', filename=f'/visualizations/gradcam_{filename}', _external=True),
            'saliency': url_for('static', filename=f'/visualizations/saliency_{filename}', _external=True),
            'lime': url_for('static', filename=f'/visualizations/lime_{filename}', _external=True),
            'gradcam_analysis': url_for('static', filename=f'/visualizations/com_analysis_{filename}', _external=True),
            "gradcam_heatmap": url_for('static', filename=f"/visualizations/{visuals['tfexplain_filename']}", _external=True),
            'prediction': predicted_class,
            'confidence': get_metric_interpretation('confidence', confidence_val),
            'patient_info': patient_info,
        }

        # 6. Generate a consolidated AI report by synthesizing all data
        # Convert complex data structures to clean, readable strings for the prompt
        patient_info_string = json.dumps(patient_info, indent=2)
        metrics_string = json.dumps(metrics, indent=2)

        # Use a simple, direct prompt for the initial image analysis
        image_analysis_prompt = "Analyze this MRI scan and provide a detailed description of your findings."

        # Get initial analyses from vision models. These will serve as inputs for the final report.
        llama3_response = get_image_description_llama3_vision(
            image_analysis_prompt, filepath)

        # medgemma_image_response = get_medical_report_from_image_medgemma(
        #     image_analysis_prompt, filepath)

        final_report_prompt_medggemma = f"""
                    Your task is to synthesize the following information from multiple sources into a single, coherent medical report.
                    ---
                    ### Source 1: AI Image Analysis (Llama3.2-vision)
                    {llama3_response}
                    ---
                    ### Source 2: Patient Information
                    {patient_info_string}
                    ---
                    ### Source 3: Quantitative Metrics
                    {metrics_string}
                    ---
                    ### Source 4: Primary AI Diagnosis
                    **Diagnosis:** {predicted_class}
                    ---
                    """

        medgemma_text_response = get_medical_report_from_text_medgemma(
            final_report_prompt_medggemma, filepath)

        # 7. Add all data to the response, including the final synthesized report
        response_data.update(metrics)

        final_report_prompt_deepsek = f"""
                    Your task is to synthesize the following information from multiple sources into a single, coherent medical report.
                    ---
                    ### Source 1: AI Image Analysis (MedGemma)
                    {medgemma_text_response}
                    ---
                    ### Source 2: AI Image Analysis (Llama3.2-vision)
                    {llama3_response}
                    ---
                    ### Source 3: Patient Information
                    {patient_info_string}
                    ---
                    ### Source 4: Quantitative Metrics
                    {metrics_string}
                    ---
                    ### Source 5: Primary AI Diagnosis
                    **Diagnosis:** {predicted_class}
                    ---
                    ### Remember:
                    - Make sure given image data description include in output. otherview student and jounir medical officers would be able to understand what you see.
                    """

        resoning_final_report = get_text_reasoning(
            final_report_prompt_deepsek, filepath)
        response_data['final_report'] = resoning_final_report

        return jsonify(response_data)

    return jsonify({'error': 'File type not allowed'}), 400


def get_metric_interpretation(metric_name, value):
    """Provides a qualitative interpretation and level for a given metric value."""
    interpretation = "N/A"
    level = "neutral"  # 'good', 'warning', 'bad'

    if metric_name == 'confidence':
        if value >= 0.9:
            interpretation = "Very High"
            level = 'good'
        elif value >= 0.7:
            interpretation = "High"
            level = 'good'
        elif value >= 0.5:
            interpretation = "Moderate"
            level = 'warning'
        else:
            interpretation = "Low"
            level = 'bad'
    elif metric_name == 'entropy':
        max_entropy = np.log2(4)  # For 4 classes
        if value <= max_entropy * 0.25:
            interpretation = "Low Uncertainty"
            level = 'good'
        elif value <= max_entropy * 0.6:
            interpretation = "Moderate Uncertainty"
            level = 'warning'
        else:
            interpretation = "High Uncertainty"
            level = 'bad'
    elif metric_name == 'margin':
        if value >= 0.7:
            interpretation = "Decisive"
            level = 'good'
        elif value >= 0.3:
            interpretation = "Reasonable"
            level = 'warning'
        else:
            interpretation = "Indecisive"
            level = 'bad'
    elif metric_name in ['dice', 'iou', 'mc_variance', 'brier']:
        # These metrics have similar good/bad ranges (high is good for dice/iou, low is good for var/brier)
        # This logic can be expanded if more specific interpretations are needed.
        pass  # Placeholder for more complex logic if needed

    return {"value": float(value), "interpretation": interpretation, "level": level}


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
@login_required
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided or invalid request format'}), 400

    message = data.get('message')
    image_name = session.get('current_image')
    if not image_name:
        return jsonify({'error': 'No image uploaded or session expired'}), 400

    prompt = f"""
                Based on the uploaded image on priviousily and the provided messages, generate answer.
                USER QUESTION: {message}
                IMAGE: {image_name}
            """

    text_response = get_text_reasoning(
        prompt, image_name)
    if text_response:
        return jsonify({'response': text_response})
    else:
        return jsonify({'error': 'Failed to get a response from the AI model.'}), 500
