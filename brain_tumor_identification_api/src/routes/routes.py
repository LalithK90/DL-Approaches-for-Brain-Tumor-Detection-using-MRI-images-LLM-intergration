from flask import Blueprint, render_template, request, jsonify, url_for, current_app, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import numpy as np
from sklearn.metrics import r2_score
from copy import deepcopy
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
from src.xai_validation.xai_metrics import (
    comprehensiveness, sufficiency, deletion_auc, insertion_auc,
    randomized_weights_test, binarize_heatmap
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
    prediction = model.predict(img_array, verbose=0)
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
    """Helper to compute all quantitative metrics including new faithfulness metrics."""
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
    dice_score_val, iou_score_val = dice_iou(gradcam_mask, lime_mask)

    # NEW: Faithfulness metrics
    class_idx = np.argmax(pred)
    mask = binarize_heatmap(cam)  # Binarize Grad-CAM for faithfulness

    comp = comprehensiveness(model, img_array[0], mask, class_idx, verbose=0)
    suff = sufficiency(model, img_array[0], mask, class_idx, verbose=0)
    del_auc = deletion_auc(
        model, img_array[0], cam, class_idx, steps=20, verbose=0)
    ins_auc = insertion_auc(
        model, img_array[0], cam, class_idx, steps=20, verbose=0)
    rand_weights = randomized_weights_test(
        model, img_array[0], class_idx, get_last_convolutional_layer(model), verbose=0)

    return {
        'top3': top3,
        'entropy': get_metric_interpretation('entropy', entropy_val),
        'margin': get_metric_interpretation('margin', margin),
        'brier': get_metric_interpretation('brier', brier),
        'center_distance': center_distance,
        'activation_ratio': activation_ratio,
        'mc_confidence_interval': mc_confidence_interval,
        'mc_variance': get_metric_interpretation('mc_variance', mc_variance),
        'dice': get_metric_interpretation('dice', dice_score_val),
        'iou': get_metric_interpretation('iou', iou_score_val),
        # NEW: Faithfulness metrics
        'comprehensiveness': get_metric_interpretation('comprehensiveness', comp),
        'sufficiency': get_metric_interpretation('sufficiency', suff),
        'deletion_auc': get_metric_interpretation('deletion_auc', del_auc),
        'insertion_auc': get_metric_interpretation('insertion_auc', ins_auc),
        'randomized_weights_corr': get_metric_interpretation('randomized_weights_corr', rand_weights),
    }

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

    elif metric_name == 'comprehensiveness':
        if value >= 0.3:
            interpretation = "Highly Faithful"
            level = 'good'
            explanation = "High comprehensiveness indicates that removing important regions significantly reduces model confidence, confirming the explanation highlights truly relevant features."
        elif value >= 0.1:
            interpretation = "Moderately Faithful"
            level = 'warning'
            explanation = "Moderate comprehensiveness suggests the explanation identifies relevant regions, but removing them only partially reduces confidence."
        else:
            interpretation = "Weakly Faithful"
            level = 'bad'
            explanation = "Low comprehensiveness indicates the explanation may not identify all critical regions, or the model relies on distributed features."

    elif metric_name == 'sufficiency':
        if value >= 0.3:
            interpretation = "Highly Sufficient"
            level = 'good'
            explanation = "High sufficiency indicates that the highlighted regions alone are enough to maintain model confidence, confirming their importance for diagnosis."
        elif value >= 0.1:
            interpretation = "Moderately Sufficient"
            level = 'warning'
            explanation = "Moderate sufficiency suggests the highlighted regions are important but the model may also use other features for accurate prediction."
        else:
            interpretation = "Weakly Sufficient"
            level = 'bad'
            explanation = "Low sufficiency indicates the highlighted regions alone cannot sustain the diagnosis, suggesting other regions also play important roles."

    elif metric_name == 'deletion_auc':
        if value >= 0.7:
            interpretation = "Excellent Deletion Sensitivity"
            level = 'good'
            explanation = "High deletion AUC indicates confidence drops rapidly as important regions are removed, confirming the explanation correctly identifies critical features."
        elif value >= 0.5:
            interpretation = "Good Deletion Sensitivity"
            level = 'warning'
            explanation = "Moderate deletion AUC suggests reasonable correspondence between highlighted regions and model decision-making."
        else:
            interpretation = "Poor Deletion Sensitivity"
            level = 'bad'
            explanation = "Low deletion AUC suggests the highlighted regions may not capture the most important features for the model's prediction."

    elif metric_name == 'insertion_auc':
        if value >= 0.7:
            interpretation = "Excellent Insertion Sensitivity"
            level = 'good'
            explanation = "High insertion AUC indicates confidence increases rapidly as important regions are inserted, confirming the explanation prioritizes truly influential features."
        elif value >= 0.5:
            interpretation = "Good Insertion Sensitivity"
            level = 'warning'
            explanation = "Moderate insertion AUC suggests the highlighted regions contribute to prediction but may require other features for full confidence."
        else:
            interpretation = "Poor Insertion Sensitivity"
            level = 'bad'
            explanation = "Low insertion AUC suggests the highlighted regions may not be the primary drivers of the model's confidence in the diagnosis."

    elif metric_name == 'randomized_weights_corr':
        if value <= 0.3:
            interpretation = "Passes Sanity Check"
            level = 'good'
            explanation = "Low correlation with randomized weights indicates the explanation genuinely depends on learned model features, passing the sanity check."
        elif value <= 0.6:
            interpretation = "Questionable Sanity"
            level = 'warning'
            explanation = "Moderate correlation suggests some dependence on model weights but also potential sensitivity to random initialization."
        else:
            interpretation = "Fails Sanity Check"
            level = 'bad'
            explanation = "High correlation with randomized weights suggests the explanation may be generating similar patterns regardless of learned model features."

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


def _generate_final_report(filepath, predicted_class, confidence_val, patient_info, metrics, llama3_response):
    patient_info_string = json.dumps(patient_info, indent=2)
    metrics_string = json.dumps(metrics, indent=2)

    final_report_prompt_medgemma = f"""
Create a comprehensive brain tumor clinical report based on this MRI image analysis. Focus on MEDICAL ASSESSMENT, education for students, and decision support for radiologists.

**AVAILABLE DATA:**
Diagnosis: {predicted_class}
Confidence: {confidence_val}
Patient Information: {patient_info_string}

**REPORT STRUCTURE (Medical-First):**

## 1. Clinical Summary (Quick Decision Support)
- What is this tumor?
- How confident are we?
- What action is needed? (STAT/Urgent/Routine)
- Key red flags if any

## 2. Patient Clinical Context
Analyze the patient data provided:
- Demographics relevant to diagnosis (age/sex epidemiology)
- Symptom correlation: Do presenting symptoms match expected pattern for {predicted_class}?
- Medical history: Any risk factors?

## 3. Imaging-Based Diagnosis
Based on the MRI you analyzed:
- Location and extent of tumor
- Key imaging features supporting {predicted_class} diagnosis
- Why this diagnosis over alternatives (brief differential)

## 4. What This Means for Patient
**Prognosis:**
- Expected outcomes with treatment
- Factors affecting prognosis for this patient specifically

**Treatment Approach:**
- Likely surgical plan based on location
- Adjuvant therapy needs (chemo/radiation)
- Timeline expectations

## 5. Educational Points for Students
- Most diagnostic features of {predicted_class}
- Common pitfalls in diagnosis
- Key management principles

**KEEP TECHNICAL AI DETAILS MINIMAL** - focus on clinical medicine, pathophysiology, and patient care.

Format: Use bullets, short paragraphs, clear headings. Be direct and educational.
    """

    medgemma_text_response = get_medical_report_from_image_medgemma(
        final_report_prompt_medgemma, filepath)

    final_report_prompt_deepsek = f"""
Synthesize a COMPLETE educational brain tumor diagnosis report using ALL available information. Follow COMMON_PROMPT_MESSAGE structure exactly, with MEDICAL CONTENT prioritized over technical details.

**DATA SOURCES TO INTEGRATE:**

1. **Image Analysis Report:** {medgemma_text_response}
2. **Patient Clinical Data:** {patient_info_string}
3. **Diagnosis:** {predicted_class}
4. **Quantitative Metrics:** {metrics_string}

**YOUR TASK:**
Create the final diagnostic report following the COMMON_PROMPT_MESSAGE structure:
- Sections 0-11: MEDICAL CONTENT (clinical decision, diagnosis, treatment, education)
- Section 12: TECHNICAL APPENDIX (AI metrics, XAI validation - for reference only)

**CRITICAL SYNTHESIS REQUIREMENTS:**

1. **Validate Consistency:**
   - Does MedGemma image analysis agree with {predicted_class}?
   - Do patient symptoms match expected presentation?
   - Do metrics support the diagnosis confidence?
   - If inconsistencies exist, address them explicitly

2. **Integrate XAI Insights INTO Clinical Sections** (not separate):
   - When describing Imaging Findings (Section 3): Mention what regions AI highlighted and why they matter clinically
   - When discussing Differential (Section 4): Use AI confidence to support likelihood percentages
   - When assessing uncertainty: Use entropy/variance to justify confidence levels

3. **Translate Metrics to Clinical Language:**
   - Comprehensiveness {metrics_string}: "Highlighted regions are [critical/somewhat relevant] to diagnosis because..."
   - Sufficiency: "These regions alone [can/cannot] explain the diagnosis, suggesting..."
   - Agreement (Dice/IoU): "Multiple AI methods [agree/disagree] on important areas, so trust level is [high/medium/low]"
   - Deletion/Insertion AUC: "Removing key regions [drastically/moderately] changes AI confidence"
   - Sanity check: "AI explanations [passed/failed] validation, meaning [trust/question] the highlighted regions"

4. **Build Evidence-Based Argument:**
   - Claim: "{predicted_class} is the diagnosis"
   - Evidence 1: Image features (from MedGemma)
   - Evidence 2: Clinical correlation (from patient data)
   - Evidence 3: Statistical confidence (from metrics)
   - Evidence 4: XAI validation (regions make clinical sense)
   - Conclusion: Confidence level with action plan

5. **Address the Dual Audience:**
   - Radiologists need: Quick summary (Section 0), confidence scores, urgency, next steps
   - Students need: Checkpoints, differential comparison table, pitfall warnings, learning pearls
   - Everyone gets: Clear medical reasoning BEFORE technical AI details

**STRUCTURAL REQUIREMENTS:**
- Use exact section numbering and titles from COMMON_PROMPT_MESSAGE
- Include ALL sections (0-12)
- Use markdown tables for differential diagnosis
- Use 🎓 for student checkpoints
- Use ⚠️ for pitfall warnings
- Medical content (0-11) comes BEFORE technical AI appendix (12)

**XAI VALIDATION APPENDIX (Section 12) - Place at END:**
Translate technical metrics to plain English:
- What regions did AI highlight? [Anatomical description]
- Do these make medical sense? [Validate against known diagnostic features]
- Trust assessment: [Based on faithfulness scores, can we trust this explanation?]
- Model performance: [Confidence, uncertainty, agreement metrics]
- Final validation: [Do all data sources converge on {predicted_class}?]

**QUALITY CHECKS:**
✓ Does Section 0 give radiologist 30-second decision summary?
✓ Does Section 4 use table format for differential?
✓ Does Section 6b include WHO grade and prognosis data?
✓ Are student checkpoints distributed throughout (not just at end)?
✓ Is technical AI content confined to Section 12 (end)?
✓ Are all claims backed by evidence from provided data sources?

**OUTPUT:** Complete clinical report following structure above, integrating all data sources into cohesive medical narrative with technical details at end.
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
    model_name = request.form.get('model_name', 'propose_balanced')
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
    response_data.update(metrics)

    final_report = _generate_final_report(
        filepath, predicted_class, confidence_val, patient_info, metrics, "not_implemented_yet")
    response_data['final_report'] = final_report

    return jsonify(response_data)



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
You are an expert neuroradiologist, neurosurgeon, and medical educator. Analyze the user's question and provide a customized, structured response that directly addresses their specific query.

**CONTEXT:**
- Case Image: {image_name}
- User has received initial AI diagnosis and clinical report
- User's Question: {message}

**STEP 1: ANALYZE THE QUESTION TYPE**

Categorize the user's question into ONE of these types:

**A. MEDICAL CONDITION INQUIRY** (Asking about disease/pathology)
   Keywords: "what is", "explain", "tell me about", "symptoms", "causes", "prognosis"
   Example: "What is glioblastoma?", "How does meningioma develop?"
   
**B. DIAGNOSTIC CLARIFICATION** (Asking about findings/diagnosis process)
   Keywords: "why diagnosed", "how did you know", "what features", "differential"
   Example: "Why was this diagnosed as pituitary adenoma?", "How is this different from meningioma?"

**C. TREATMENT & MANAGEMENT** (Asking about clinical decisions)
   Keywords: "treatment", "surgery", "therapy", "management", "what should", "next steps"
   Example: "What surgery is needed?", "Should this get radiation?"

**D. IMAGING INTERPRETATION** (Asking about MRI/radiology specifics)
   Keywords: "MRI shows", "what does this mean", "imaging findings", "scan"
   Example: "What does the bright area mean?", "Why does it enhance?"

**E. EDUCATIONAL/LEARNING** (Student wanting to learn/understand concepts)
   Keywords: "how does", "why", "mechanism", "explain the process", "teach me"
   Example: "How does brain herniation occur?", "Why do tumors cause seizures?"

**F. AI/TECHNICAL EXPLANATION** (Asking about the AI system)
   Keywords: "AI", "model", "confidence", "algorithm", "how computer", "metrics"
   Example: "What does 95% confidence mean?", "How does Grad-CAM work?"

**G. PROGNOSTIC INQUIRY** (Asking about outcomes/future)
   Keywords: "survival", "outcome", "prognosis", "cure rate", "life expectancy"
   Example: "What is the survival rate?", "Can this be cured?"

**H. COMPARISON REQUEST** (Asking to compare conditions/options)
   Keywords: "difference between", "compare", "versus", "which is better"
   Example: "Difference between glioma and meningioma?", "MRI vs CT scan?"

---

**STEP 2: STRUCTURE YOUR RESPONSE BASED ON QUESTION TYPE**

**FOR TYPE A - MEDICAL CONDITION INQUIRY:**

**[CONDITION NAME]**

**Quick Definition:**
[1-2 sentence simple explanation]

**Key Characteristics:**
• What it is (pathology)
• Where it occurs (location/anatomy)
• Who gets it (epidemiology: age, sex, risk factors)
• How common (incidence/prevalence)

**Clinical Presentation:**
• Main symptoms
• How it's discovered
• Typical patient profile

**Diagnosis:**
• Imaging features (MRI/CT findings)
• Key diagnostic criteria
• How it's confirmed (biopsy/pathology)

**Management Overview:**
• Treatment approach
• Expected outcomes
• Prognosis factors

🎓 **Student Pearl:** [Key learning point or memory aid]

📚 **Learning Resources for This Condition:**
• **Radiopaedia**: Search "[condition name] imaging" - https://radiopaedia.org/
• **StatPearls**: Free textbook chapter on [condition] - https://www.ncbi.nlm.nih.gov/books/
• **UpToDate**: Clinical overview - https://www.uptodate.com/
• **Suggested Keywords**: "[condition]", "[key imaging feature]", "[differential diagnosis terms]"

---

**FOR TYPE B - DIAGNOSTIC CLARIFICATION:**

**WHY THIS DIAGNOSIS: [Diagnosis Name]**

**Direct Answer:**
[2-3 sentences explaining the diagnosis based on this case]

**Supporting Evidence from This Case:**
1. **Imaging Features:**
   • [Specific finding 1 from MRI]
   • [Specific finding 2 from MRI]
   • [Location/characteristics]

2. **Clinical Correlation:**
   • [How symptoms match]
   • [Age/sex epidemiology fit]

3. **AI Analysis:**
   • Confidence: [X%] - meaning [interpretation]
   • Key regions identified: [anatomical areas]

**Differential Diagnosis:**
| Diagnosis | Likelihood | Why Considered | Why Ruled Out/Less Likely |
|-----------|------------|----------------|---------------------------|
| [This diagnosis] | High | [features supporting] | - |
| [Alternative 1] | Low | [similar features] | [distinguishing factors] |
| [Alternative 2] | Low | [similar features] | [distinguishing factors] |

**Confidence Assessment:**
[Explain certainty level and what would increase/decrease it]

---

**FOR TYPE C - TREATMENT & MANAGEMENT:**

**CLINICAL MANAGEMENT PLAN**

**Immediate Actions:**
• [Urgent/STAT interventions if needed]
• [Timeframe for action]
• [Monitoring requirements]

**Treatment Strategy:**

1. **Primary Treatment:**
   • [Surgery/radiation/medical - specifics]
   • Rationale: [why this approach]
   • Timing: [when to proceed]

2. **Adjuvant Therapy:**
   • [Additional treatments needed]
   • Sequence and duration

3. **Supportive Care:**
   • [Symptom management]
   • [Quality of life measures]

**Decision Factors:**
• [What influences treatment choice]
• [Patient-specific considerations]
• [Risk-benefit analysis]

**Expected Timeline:**
[Treatment duration and follow-up schedule]

⚠️ **Important:** [Key warnings or considerations]

**Guidelines Reference:** [Cite NCCN/WHO/relevant guidelines]

---

**FOR TYPE D - IMAGING INTERPRETATION:**

**IMAGING FINDINGS EXPLAINED**

**What You're Seeing:**
[Simple description of the finding in question]

**Clinical Significance:**
• **Normal vs Abnormal:** [Establish baseline]
• **What It Indicates:** [Pathophysiology]
• **Why It Matters:** [Clinical implications]

**Technical Explanation:**

**For Radiologists:**
[Detailed technical interpretation with proper terminology]

**For Students/Non-Radiologists:**
[Simplified explanation with analogies]
Example: [Use relatable comparison]

**Correlation with Diagnosis:**
[How this finding supports/confirms the diagnosis]

**Additional Imaging Considerations:**
• [Other sequences that might help]
• [What to look for in follow-up]

---

**FOR TYPE E - EDUCATIONAL/LEARNING:**

**🎓 TEACHING MODULE: [Topic]**

**Level 1 - Simple Explanation:**
[Explain in simple terms anyone can understand]

**Level 2 - Mechanism/Process:**
[Detailed explanation of how/why it works]

**Step-by-Step:**
1. [Process step 1]
2. [Process step 2]
3. [Process step 3]

**Visual/Spatial Understanding:**
[Describe anatomy/pathology in 3D context]

**Clinical Application:**
[How this concept applies to real patient care]

**Common Misconceptions:**
❌ Wrong: [Common mistake]
✅ Correct: [Accurate understanding]

**Memory Aid:**
💡 [Mnemonic or memory trick]

**Practice Questions:**
• [Self-test question 1]
• [Self-test question 2]

**Next Learning Steps:**
📚 Read about: [Related topic to study next]
🔍 Look up: [Specific cases/examples]

---

**📖 RECOMMENDED LEARNING RESOURCES:**

**Online Medical Resources:**
1. **Radiopaedia** - https://radiopaedia.org/
   • Search for: "[specific topic keywords]"
   • Best for: Imaging examples with annotated cases
   • Why useful: Visual learning with real MRI examples

2. **Neurosurgical Atlas** - https://www.neurosurgicalatlas.com/
   • Search for: "[tumor type] or [surgical approach]"
   • Best for: Surgical anatomy and treatment approaches
   • Why useful: Detailed operative videos and 3D anatomy

3. **UpToDate** - https://www.uptodate.com/
   • Search for: "[condition] pathophysiology" or "[condition] management"
   • Best for: Evidence-based clinical guidelines
   • Why useful: Comprehensive, regularly updated clinical information

4. **StatPearls (NCBI)** - https://www.ncbi.nlm.nih.gov/books/NBK430685/
   • Free medical textbook chapters
   • Best for: Quick reference and board exam prep
   • Why useful: Concise, peer-reviewed content

5. **PubMed** - https://pubmed.ncbi.nlm.nih.gov/
   • Search query: "[specific keywords] AND (review OR meta-analysis)"
   • Best for: Latest research and systematic reviews
   • Why useful: Access to cutting-edge studies

6. **Osmosis** - https://www.osmosis.org/
   • Video-based learning platform
   • Best for: Visual learners and quick concept reviews
   • Why useful: Animated explanations of complex concepts

**Suggested Search Terms for This Topic:**
• "[Primary search term related to question]"
• "[Secondary search term for deeper dive]"
• "[Comparison term if differential diagnosis relevant]"

**Related Topics to Study:**
• [Related topic 1] - Helps understand: [connection to main topic]
• [Related topic 2] - Important because: [clinical relevance]
• [Related topic 3] - Builds foundation for: [advanced concept]

**Clinical Practice Resources:**
• **NCCN Guidelines** - https://www.nccn.org/guidelines/category_1
  For: Treatment algorithms and decision support
• **WHO Classification** - For: Tumor grading and classification criteria
• **Case Reports** - Search PubMed for "[condition] case report" for real-world examples

**Why These Resources Matter:**
- Radiopaedia/Neurosurgical Atlas → Visual pattern recognition
- UpToDate/StatPearls → Clinical decision-making
- PubMed → Understanding latest evidence
- Osmosis → Conceptual understanding and retention

💡 **Learning Tip:** Start with Radiopaedia for imaging, then UpToDate for management, finally PubMed for latest research.

---

**FOR TYPE F - AI/TECHNICAL EXPLANATION:**

**AI SYSTEM EXPLANATION (Non-Technical)**

**What You Asked About:** {message}

**Simple Answer:**
[Explain in everyday language what the AI metric/feature means]

**Clinical Translation:**
Think of it like: [Medical analogy]

**What It Means for This Case:**
• [Specific interpretation for this patient]
• [Confidence/reliability level]
• [How to use this information]

**Technical Details (Optional):**
<details>
<summary>For those interested in the technical side</summary>

[More detailed technical explanation with proper terms]

**How It Works:**
1. [Process step 1]
2. [Process step 2]

**Validation:**
[How we know it's reliable]
</details>

**Practical Takeaway:**
[What the user should actually do with this information]

---
## External Learning Resources & Online References

**For Students & Radiologists to Study Further:**

### Online Atlases & Visual References
- **Neurosurgical Atlas** (https://www.neurosurgicalatlas.com/)
  * Search for: [Tumor type, anatomy, surgical approaches]
  * Why useful: High-quality illustrations of normal anatomy, tumor pathology, surgical corridors
  
- **Radiopaedia** (https://radiopaedia.org/)
  * Search: "[Tumor type] imaging", "[Tumor type] MRI findings"
  * Why useful: Large database of cases with imaging features, differential diagnosis, clinical correlation
  
- **Osmosis** (https://www.osmosis.org/)
  * Search: "[Tumor type] brain tumor", "[Diagnosis] pathophysiology"
  * Why useful: Student-focused, explains pathology and clinical manifestations clearly

- **UpToDate** (https://www.uptodate.com/)
  * Search: "[Tumor type] epidemiology", "[Tumor type] diagnosis and management"
  * Why useful: Comprehensive, regularly updated clinical information (subscription required)

### Clinical Guidelines & Protocols
- **NCCN Guidelines** (https://www.nccn.org/professionals/physician_gls/pdf/cns.pdf)
  * For: Treatment protocols, staging, surveillance schedules
  * Latest version: [Current year]
  
- **WHO Classification** (https://www.who.int/publications/item/9789240045681)
  * For: Tumor grading criteria, molecular markers, prognostic groups
  
- **EANO Guidelines** (https://www.eano.eu/)
  * For: European standards, brain tumor management in different patient populations

### Medical Imaging & Diagnostic References
- **Osborn's Brain** (https://www.elsevier.com/books/osborns-brain/osborn/)
  * Gold standard textbook for neuroimaging - covers all brain tumors with imaging patterns
  
- **Diagnostic Imaging: Brain** (Elsevier)
  * 2000+ high-quality cases with clinical correlation
  
- **American Journal of Neuroradiology** (https://www.ajnr.org/)
  * Latest research on imaging techniques for tumor diagnosis and follow-up

### Pathophysiology & Deep Dive Learning
- **Johns Hopkins Brain Tumor Center** (https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumors)
  * Information for: Patient education, clinical staging, prognosis
  
- **Mayo Clinic Brain Tumor Resources** (https://www.mayoclinic.org/diseases-conditions/brain-tumor/)
  * For: Symptoms, diagnosis, staging, treatment options
  
- **National Brain Tumor Society** (https://www.braintumorconnect.org/)
  * For: Patient support, latest research, clinical trials, community resources

### Research & Latest Evidence
- **PubMed** (https://pubmed.ncbi.nlm.nih.gov/)
  * Search: "[Tumor type] epidemiology [Current Year]", "[Tumor type] prognosis"
  * Find: Latest peer-reviewed publications on this tumor type
  
- **Google Scholar** (https://scholar.google.com/)
  * Search: "[Tumor type] imaging features", "[Tumor type] management"
  * Access: Full-text papers where available
  
- **ResearchGate** (https://www.researchgate.net/)
  * Search: "[Tumor type]", "[Your tumor type] MRI"
  * Connect: With researchers studying this tumor type

### Student Learning Platforms
- **Khan Academy** (https://www.khanacademy.org/)
  * Search: "Brain anatomy", "Nervous system", "Cancer biology"
  * Why: Foundation knowledge for tumor pathophysiology

- **Lecturio** (https://www.lecturio.com/)
  * Search: "Brain tumors", "Neurooncology", "Surgical neuropathology"
  * Why: Video-based learning, organized by specialty

### Related Clinical Conditions to Consider
**Sometimes brain imaging findings relate to other diagnoses:**
- **Demyelinating disease:** MS, ADEM (can mimic tumors)
- **Infectious diseases:** Abscess, toxoplasmosis (especially in immunocompromised)
- **Vascular disorders:** Cavernoma, AVM, aneurysm (mass-like lesions)
- **Inflammatory conditions:** Sarcoidosis, vasculitis (can present as masses)
- **Metabolic disorders:** Leukodystrophies (diffuse changes vs focal mass)

**Resources for these differentials:**
- https://radiopaedia.org/articles/differential-diagnosis-of-intracranial-masses
- https://www.osmosis.org/ (search by symptom or imaging finding)

### Red Flags & Urgent Learning Points
**When to recognize and escalate:**
- **Imaging features requiring STAT action:** [Herniating mass, acute hemorrhage, etc.]
- **Clinical scenarios requiring immediate intervention:** [Seizure status, rapidly progressive deficit, etc.]
- **Resource:** https://www.neurosurgerytoday.org/ (peer-reviewed neurosurgery news)

---

**HOW TO USE THESE RESOURCES:**
1. **Students:** Start with Khan Academy/Osmosis for basics, then move to Radiopaedia for cases
2. **Radiologists:** Use Radiopaedia to compare cases, NCCN for staging, UpToDate for management updates
3. **Clinicians:** Reference NCCN/WHO for protocols, Mayo/Johns Hopkins for patient counseling
4. **All Professionals:** Check PubMed/Google Scholar for latest research on THIS specific tumor type

---
**FOR TYPE G - PROGNOSTIC INQUIRY:**

**PROGNOSIS & OUTCOMES**

**Direct Answer:**
[Clear statement about expected outcomes]

**Survival Statistics:**
| Timeframe | Survival Rate | Notes |
|-----------|---------------|-------|
| 5-year | [X%] | [Context] |
| 10-year | [X%] | [Context] |

**Factors Affecting Prognosis:**

**Favorable Factors:**
✓ [Factor 1]
✓ [Factor 2]

**Unfavorable Factors:**
⚠️ [Factor 1]
⚠️ [Factor 2]

**For This Specific Patient:**
[Personalized prognosis based on available case data]

**Treatment Impact:**
• Without treatment: [Expected outcome]
• With standard treatment: [Expected outcome]
• Best case scenario: [Conditions needed]

**Quality of Life Considerations:**
[Expected functional outcomes and life impact]

**Important Context:**
[Statistics limitations, individual variation, hope vs realism]

---

**FOR TYPE H - COMPARISON REQUEST:**

**COMPARISON: [Condition A] vs [Condition B]**

| Feature | [Condition A] | [Condition B] |
|---------|---------------|---------------|
| **Definition** | [Brief description] | [Brief description] |
| **Location** | [Where it occurs] | [Where it occurs] |
| **Age/Sex** | [Demographics] | [Demographics] |
| **Imaging** | [MRI appearance] | [MRI appearance] |
| **Symptoms** | [Clinical presentation] | [Clinical presentation] |
| **Treatment** | [Management approach] | [Management approach] |
| **Prognosis** | [Outcomes] | [Outcomes] |

**Key Distinguishing Features:**
🔍 [Most important difference that helps tell them apart]

**Overlapping Features:**
⚠️ [What they have in common - causes diagnostic confusion]

**Clinical Decision Impact:**
[Why the distinction matters for patient care]

**For This Case:**
[Which one applies and why]

---

**STEP 3: QUALITY CHECKS**

Before finalizing your response:
✓ Did you directly answer the user's specific question in first 2 sentences?
✓ Is the structure appropriate for the question type?
✓ Did you use formatting (bullets, tables, bold) for clarity?
✓ Is technical jargon explained or avoided?
✓ Did you reference the specific case/image when relevant?
✓ Is the response educational but not overwhelming?
✓ Did you maintain medical accuracy?
✓ Did you acknowledge limitations if applicable?

**ALWAYS END WITH:**
---
**Related Questions You Might Ask:**
• [Suggested follow-up question 1]
• [Suggested follow-up question 2]
• [Suggested follow-up question 3]
• [Suggested follow-up question 4]

💬 Feel free to ask for clarification or dive deeper into any aspect!
            """
    text_response = get_text_reasoning(prompt, image_name)
    if text_response:
        return jsonify({'response': text_response})
    else:
        return jsonify({'error': 'Failed to get a response from the AI model.'}), 500


api_blueprint = Blueprint('api', __name__)


@api_blueprint.route('/predict', methods=['POST'])
def predict():
    # Validate file upload
    file, error_response, status_code = _validate_file_upload(request)
    if error_response:
        return error_response, status_code

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    filepath = _save_uploaded_file(file, filename)
    session['current_image'] = filename.replace(" ", "")
    current_image = filename.replace(" ", "")
    print(f"File saved at: {current_image}")
    # Process the image and load the model
    model_name = request.form.get('model_name', 'propose_balanced')
    img, img_array, model = _process_image(filepath, model_name)
    if model is None:
        return jsonify({'error': 'Model not found'}), 404

    # Get model prediction
    pred, class_idx, predicted_class, confidence_val = _process_and_predict(
        model, img_array)

    # Generate and save XAI visualizations
    try:
        visuals = _generate_visualizations(
            model, img_array, class_idx, filename)
        _save_visualizations(visuals, img, filename)
    except ValueError as e:
        return jsonify({'error': str(e)})

    # Calculate metrics
    metrics = _calculate_metrics(
        pred, visuals['cam'], visuals['lime_img'], model, img_array, LABELS, filename)

    # Find matching patient data
    with open(current_app.config['PATIENT_DATA_PATH'], 'r') as f:
        patient_data = json.load(f)
    patient_info = find_patient_by_diagnosis(predicted_class, patient_data)
    del patient_info["diagnosis"]

    # Assemble the response data
    response_data = _generate_response_data(
        filename, predicted_class, confidence_val, patient_info, visuals)
    response_data.update(metrics)

    final_report = _generate_final_report(
        filepath, predicted_class, patient_info, metrics, "not_implemented_yet")
    response_data['final_report'] = final_report

    return jsonify(response_data)
