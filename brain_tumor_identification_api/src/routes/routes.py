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

        # 6. Assemble and return the final JSON response with educational explanations
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
            'xai_educational_notes': {
                'gradcam_explanation': "Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions in the image that most strongly influenced the model's prediction. Brighter areas indicate features that strongly support the diagnosis.",
                'saliency_explanation': "Saliency maps show which pixels in the image had the greatest influence on the classification decision. They help identify the specific image features the model focused on.",
                'lime_explanation': "Local Interpretable Model-agnostic Explanations (LIME) creates a simplified local model to explain which features contributed most to the prediction. It helps understand which image regions support or contradict the diagnosis.",
                'metrics_explanation': "Quantitative metrics provide objective measures of model confidence and uncertainty. They help assess the reliability of the AI diagnosis and identify cases that may require additional clinical correlation."
            }
        }

        # 6. Generate a consolidated AI report by synthesizing all data
        # Convert complex data structures to clean, readable strings for the prompt
        patient_info_string = json.dumps(patient_info, indent=2)
        metrics_string = json.dumps(metrics, indent=2)

        # Use a detailed prompt for the initial image analysis that focuses on specific radiological features
        image_analysis_prompt = """Analyze this brain MRI scan with the precision of an expert neuroradiologist. Provide a detailed description of your findings, including:
1. Precise anatomical localization of any abnormalities
2. Signal characteristics on different sequences (if visible)
3. Size measurements and shape description of any lesions
4. Mass effect, edema, or midline shift
5. Enhancement patterns (if contrast is visible)
6. Involvement of critical structures
7. Secondary findings or complications
8. Specific radiological signs that suggest a particular diagnosis

Be thorough in your analysis and use precise medical terminology while explaining key concepts."""

        # Get initial analyses from vision models. These will serve as inputs for the final report.
        llama3_response = get_image_description_llama3_vision(
            image_analysis_prompt, filepath)

        # medgemma_image_response = get_medical_report_from_image_medgemma(
        #     image_analysis_prompt, filepath)

        final_report_prompt_medggemma = f"""
                    Your task is to synthesize the following information from multiple sources into a comprehensive, educational medical report for brain tumor diagnosis.
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
                    
                    Structure your report with these sections:
                    
                    1. **Clinical Presentation Summary**: Synthesize patient demographics, symptoms, and relevant history.
                    
                    2. **Imaging Findings**: Describe key observations with precise anatomical references and measurements. Include both primary features (tumor characteristics) and secondary features (mass effect, edema, etc.).
                    
                    3. **Quantitative Assessment**: Interpret the AI confidence metrics in clinical terms. Explain what the entropy, margin, and other metrics suggest about diagnostic certainty.
                    
                    4. **Differential Diagnosis**: Present a ranked list of possible diagnoses with specific imaging features supporting or contradicting each. For the primary diagnosis, provide a detailed explanation of pathognomonic features.
                    
                    5. **Pathophysiology Insights**: Explain the underlying disease mechanisms and cellular characteristics of the identified tumor type.
                    
                    6. **Clinical Implications**: Discuss expected symptoms, progression patterns, and potential complications.
                    
                    7. **Management Recommendations**: Suggest evidence-based treatment approaches with their mechanisms of action and expected outcomes.
                    
                    8. **Educational Pearls**: Include 3-5 high-yield learning points that highlight critical diagnostic features and common pitfalls in diagnosing this type of tumor.
                    
                    9. **References**: Mention relevant clinical guidelines or landmark studies.
                    
                    Use precise medical terminology while providing clear explanations for complex concepts. Your goal is to both accurately diagnose the current case and enhance the medical practitioner's knowledge for future cases.
                    """

        medgemma_text_response = get_medical_report_from_text_medgemma(
            final_report_prompt_medggemma, filepath)

        # 7. Add all data to the response, including the final synthesized report
        response_data.update(metrics)

        final_report_prompt_deepsek = f"""
                    Your task is to synthesize the following information from multiple sources into a comprehensive, educational medical report for brain tumor diagnosis. You are creating the FINAL report that will be presented to medical practitioners, so ensure it is complete, accurate, and maximally educational.
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
                    ### Source 6: Explainable AI Visualizations
                    The following visualization techniques were applied to understand the model's decision-making process:
                    - Grad-CAM: Highlights regions most influential in the classification decision
                    - Saliency Map: Shows pixels that most strongly influence the prediction
                    - LIME: Provides local interpretable explanations of model predictions
                    
                    These visualizations indicate which regions of the image were most important for the AI's diagnosis. Use this information to explain which specific imaging features were most significant in determining the diagnosis.
                    ---
                    
                    Structure your report with these sections:
                    
                    1. **Executive Summary**: A concise overview of the case, diagnosis, and key findings (2-3 sentences).
                    
                    2. **Clinical Presentation**: Synthesize patient demographics, symptoms, and relevant history.
                    
                    3. **Imaging Findings**: Describe key observations with precise anatomical references and measurements. Include both primary features (tumor characteristics) and secondary features (mass effect, edema, etc.). Explain which specific regions were highlighted by the XAI visualizations and their significance.
                    
                    4. **Quantitative Assessment**: Interpret the AI confidence metrics in clinical terms. Explain what the entropy, margin, and other metrics suggest about diagnostic certainty. Discuss any discrepancies between different metrics and their implications.
                    
                    5. **Differential Diagnosis**: Present a ranked list of possible diagnoses with specific imaging features supporting or contradicting each. For the primary diagnosis, provide a detailed explanation of pathognomonic features.
                    
                    6. **Pathophysiology Insights**: Explain the underlying disease mechanisms, cellular characteristics, and molecular markers of the identified tumor type.
                    
                    7. **Clinical Implications**: Discuss expected symptoms, progression patterns, and potential complications. Include information about typical prognosis based on the specific features observed.
                    
                    8. **Management Recommendations**: Suggest evidence-based treatment approaches with their mechanisms of action and expected outcomes. Include surgical considerations, radiation therapy options, and chemotherapy protocols when applicable.
                    
                    9. **Educational Pearls**: Include 3-5 high-yield learning points that highlight critical diagnostic features, common pitfalls, and distinguishing characteristics from similar conditions. Use clear comparisons and analogies where helpful.
                    
                    10. **References**: Mention relevant clinical guidelines or landmark studies that inform the diagnostic and treatment approach.
                    
                    Important guidelines:
                    - Use precise medical terminology while providing clear explanations for complex concepts.
                    - Include detailed image descriptions so students and junior medical officers can understand what they're seeing.
                    - Structure your report with clear headings and bullet points for readability.
                    - Highlight key findings and critical information using bold or italics where appropriate.
                    - Connect imaging findings to clinical presentation and expected symptoms.
                    - Explain the significance of the XAI visualizations in understanding the diagnosis.
                    - Your goal is to both accurately diagnose the current case and enhance the medical practitioner's knowledge for future cases.
                    """

        resoning_final_report = get_text_reasoning(
            final_report_prompt_deepsek, filepath)
        response_data['final_report'] = resoning_final_report

        return jsonify(response_data)

    return jsonify({'error': 'File type not allowed'}), 400


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
                You are an expert neuroradiologist and neurosurgeon specializing in brain tumors. A medical practitioner is asking you a question about a brain MRI scan they've uploaded. Your goal is to provide a detailed, educational response that not only answers their specific question but also enhances their understanding of brain tumor diagnosis and management.
                
                When answering, follow these guidelines:
                1. Directly address the specific question asked
                2. Provide detailed explanations with anatomical precision
                3. Include relevant pathophysiological mechanisms when applicable
                4. Reference specific imaging features visible in the scan
                5. Mention relevant clinical correlations
                6. Add educational insights that go beyond the immediate question
                7. Use precise medical terminology while explaining complex concepts
                8. When appropriate, mention recent advances or guidelines in the field
                
                USER QUESTION: {message}
                IMAGE: {image_name}
                
                Remember to structure your response clearly with appropriate headings and bullet points when needed. Your goal is to be both informative and educational, helping the medical practitioner improve their diagnostic skills and knowledge.
            """

    text_response = get_text_reasoning(
        prompt, image_name)
    if text_response:
        return jsonify({'response': text_response})
    else:
        return jsonify({'error': 'Failed to get a response from the AI model.'}), 500
