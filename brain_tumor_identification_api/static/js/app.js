document.addEventListener('DOMContentLoaded', function () {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach(function (tooltipTriggerEl) {
        new bootstrap.Tooltip(tooltipTriggerEl);
    });
    document.getElementById('uploadButton').addEventListener('click', uploadImage);
});

function getValueColor (value, min, max, reverse = false) {
    // Clamp the value to be within min and max for accurate color mapping
    const clampedValue = Math.max(min, Math.min(value, max));
    let normalized = (clampedValue - min) / (max - min);

    if (reverse) {
        normalized = 1 - normalized;
    }

    // Hue: 0 is red, 60 is yellow, 120 is green
    const hue = normalized * 120;
    // Use HSL for an easy and vibrant color transition
    return `hsl(${hue}, 90%, 45%)`;
}

async function uploadImage () {
    const uploadButton = document.getElementById('uploadButton');
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) {
        alert('Please select an image');
        return;
    }

    // Hide patient card and reset table styles on new upload
    document.getElementById('patient-info-card').classList.add('d-none');
    const tableCells = document.querySelectorAll('#metrics-table-container td');
    tableCells.forEach(cell => {
        cell.textContent = '-';
        cell.style.backgroundColor = 'transparent';
        cell.style.color = 'inherit';
        cell.style.fontWeight = 'normal';
    });

    uploadButton.disabled = true;
    input.disabled = true;
    uploadButton.innerHTML = `
    <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
    Analyzing...
    `;


    const formData = new FormData();
    formData.append('file', file);

    const modelName = document.getElementById('modelSelect').value;
    formData.append('model_name', modelName);

    try {
        const predictUrl = uploadButton.dataset.predictUrl;
        const response = await fetch(predictUrl, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        console.log(data);

        if (data.error) {
            alert(data.error);
            return;
        }

        if (!data.original) {
            document.getElementById('result_images').classList.add('d-none');
        } else {
            document.getElementById('result_images').classList.remove('d-none');
            // Use direct URLs from JSON
            document.getElementById('originalImg').src = data.original;
            document.getElementById('gradcamImg').src = data.gradcam;
            document.getElementById('saliencyImg').src = data.saliency;
            document.getElementById('limeImg').src = data.lime;
            document.getElementById('gradcamAnalysisImg').src = data.gradcam_analysis;
            document.getElementById('gradcamHeatmapImg').src = data.gradcam_heatmap;
        }

        // --- Populate and Color Metrics Table ---
        const maxEntropy = Math.log2(4); // For 4 classes
        const maxCenterDist = Math.sqrt(Math.pow(224 / 2, 2) * 2); // Max distance from center

        const setMetricCell = (id, value, displayValue, min, max, reverse = false) => {
            const cell = document.getElementById(id);
            cell.textContent = displayValue;
            if (value !== null && !isNaN(value)) {
                cell.style.backgroundColor = getValueColor(value, min, max, reverse);
                cell.style.color = 'white';
                cell.style.fontWeight = 'bold';
            }
        };
        if (!data.prediction) {
            document.getElementById('metrics-table-container').classList.add('d-none');
            document.getElementById('accordionPanelsStayOpenExample').classList.add('d-none');
            document.getElementById('chatSection').classList.add('d-none');
        } else {
            // Add initial chat message
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML = `
                <div class="d-flex flex-row justify-content-start mb-3">
                    <div class="p-3 ms-3" style="border-radius: 15px; background-color: #f0f0f0;">
                        <p class="small mb-0">
                            <strong class="text-success"><i class="fas fa-robot me-2"></i>RadioAI:</strong>
                            Hello! The analysis is complete. Feel free to ask me any questions about the results.
                        </p>
                    </div>
                </div>`;
            document.getElementById('metrics-table-container').classList.remove('d-none');
            document.getElementById('accordionPanelsStayOpenExample').classList.remove('d-none');
            document.getElementById('chatSection').classList.remove('d-none');
        }

        // Table metrics
        document.getElementById('prediction-value').textContent = data.prediction;
        document.getElementById('top3-value').textContent = data.top3.map(
            t => `${t[0]}: ${(t[1] * 100).toFixed(2)}%`
        ).join(', ');
        document.getElementById('activation-ratio-value').textContent = (data.activation_ratio * 100).toFixed(2) + "%";
        document.getElementById('mc-ci-value').textContent = `[${data.mc_confidence_interval[0].toFixed(2)}, ${data.mc_confidence_interval[1].toFixed(2)}]`;

        setMetricCell('confidence-value', data.confidence.value, (data.confidence.value * 100).toFixed(2) + "%", 0, 1);
        setMetricCell('center-distance-value', data.center_distance, data.center_distance.toFixed(2), 0, maxCenterDist, true);
        setMetricCell('dice-value', data.dice.value, (data.dice.value * 100).toFixed(2) + "%", 0, 1);
        setMetricCell('iou-value', data.iou.value, (data.iou.value * 100).toFixed(2) + "%", 0, 1);
        setMetricCell('entropy-value', data.entropy.value, data.entropy.value.toFixed(3), 0, maxEntropy, true);
        setMetricCell('margin-value', data.margin.value, data.margin.value.toFixed(3), 0, 1);
        setMetricCell('mc-variance-value', data.mc_variance.value, data.mc_variance.value.toFixed(3), 0, 0.25, true);
        setMetricCell('brier-value', data.brier.value, data.brier.value.toFixed(3), 0, 1, true);

        // Populate Patient Info Card if data exists
        const patientInfo = data.patient_info;
        if (patientInfo) {
            document.getElementById('patient-info-card').classList.remove('d-none');
        } else {
            document.getElementById('patient-info-card').classList.add('d-none');
        }
        showPatientInfo(patientInfo);

        const aiResponseHtml = `
                <div class="d-flex flex-row justify-content-start mb-3">
                    <div class="p-3 ms-3" style="border-radius: 15px; background-color: #f0f0f0;">
                        ${convertMarkdownToText(data.final_report)}
                    </div>
                </div>`;
        $('#chatBox').append(aiResponseHtml);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during upload');
    } finally {
        uploadButton.disabled = false;
        input.disabled = false;
        uploadButton.innerHTML = 'Upload & Analyze';
    }
}
function showPatientInfo (patientInfo) {
    const setText = (id, value) => {
        document.getElementById(id).textContent = value || 'N/A';
    };

    if (patientInfo) {
        setText('patient-id', patientInfo.patient_id);
        setText('patient-name', patientInfo.patient_description.demographics.name);
        setText('patient-age', patientInfo.patient_description.demographics.age);
        setText('patient-gender', patientInfo.patient_description.demographics.gender);
        setText('patient-ethnicity', patientInfo.patient_description.demographics.ethnicity);
        setText('patient-height', patientInfo.patient_description.vitals.height_cm);
        setText('patient-weight', patientInfo.patient_description.vitals.weight_kg);
        setText('patient-bmi', patientInfo.patient_description.vitals.bmi);
        setText('patient-bp', patientInfo.patient_description.vitals.blood_pressure);
        setText('patient-pulse', patientInfo.patient_description.vitals.pulse);
        setText('patient-occupation', patientInfo.patient_description.lifestyle.occupation);
        setText('patient-smoking', patientInfo.patient_description.lifestyle.smoking_status);
        setText('patient-alcohol', patientInfo.patient_description.lifestyle.alcohol_use);
        setText('patient-exercise', patientInfo.patient_description.lifestyle.exercise_frequency);
        setText('patient-last-visit', patientInfo.patient_description.last_physician_visit);
        setText('patient-mri-indication', patientInfo.patient_description.mri_indication || 'Not specified');

        const populateList = (elementId, items, formatter, emptyText) => {
            const listElement = document.getElementById(elementId);
            listElement.innerHTML = '';
            if (items && items.length > 0) {
                items.forEach(item => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = formatter(item);
                    listElement.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.textContent = emptyText;
                listElement.appendChild(li);
            }
        };

        populateList('patient-symptoms', patientInfo.symptoms, s => `${s.description} (Severity: ${s.severity || 'N/A'}, Onset: ${s.onset || 'N/A'})`, 'No specific symptoms listed.');
        populateList('patient-past-diagnoses', patientInfo.medical_history?.past_diagnoses, d => `${d.condition} (Duration: ${d.duration_years || 'N/A'} years)`, 'No past diagnoses listed.');
        populateList('patient-family-history', patientInfo.medical_history?.family_history, h => `${h.relation}: ${h.condition}`, 'No family history listed.');
        populateList('patient-surgeries', patientInfo.medical_history?.surgeries, s => `${s.procedure} (Year: ${s.year || 'N/A'})`, 'No surgeries listed.');

        setText('patient-medications', patientInfo.medical_history?.medications?.join(', ') || 'None');
        setText('patient-allergies', patientInfo.medical_history?.allergies?.join(', ') || 'None');
        setText('patient-mri-modality', patientInfo.imaging?.mri_modality);
        setText('patient-scan-date', patientInfo.imaging?.scan_date);
        setText('patient-findings', patientInfo.imaging?.findings_summary);

        document.getElementById('patient-info-card').classList.remove('d-none');
    }
};


$('#sendChat').on('click', function () {
    startProgressAlert();
    const message = $('#chatInput').val();
    if (message.trim() === '') {
        endProgressAlert(); // Close the alert if message is empty
        return;
    }

    // Sanitize user message to prevent HTML injection
    const sanitizedMessage = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // User's message
    const userMessageHtml = `
        <div class="d-flex flex-row justify-content-end mb-3">
            <div class="p-3 me-3 border" style="border-radius: 15px; background-color: #e6f7ff;">
                <p class="small mb-0">
                    <strong class="text-primary"><i class="fas fa-user-doctor me-2"></i>You:</strong>
                    ${sanitizedMessage}
                </p>
            </div>
        </div>`;
    $('#chatBox').append(userMessageHtml);
    $('#chatInput').val('');
    $('#chatBox').animate({ scrollTop: $('#chatBox')[0].scrollHeight }, 300);


    const formData = {
        message: message,
    };
    $.ajax({
        url: '/chat',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(formData),
        success: function (response) {
            endProgressAlert();
            const aiResponseHtml = `
                <div class="d-flex flex-row justify-content-start mb-3">
                    <div class="p-3 ms-3" style="border-radius: 15px; background-color: #f0f0f0;">
                        ${convertMarkdownToText(response.response)}
                    </div>
                </div>`;
            $('#chatBox').append(aiResponseHtml);
            $('#chatBox').animate({
                scrollTop: $('#chatBox')[0].scrollHeight
            }, 300);
        },
        error: function (xhr) {
            endProgressAlert();
            console.error('Error saving additional data:', xhr.responseText);
            const errorMessage = `
                <div class="d-flex flex-row justify-content-start mb-3">
                     <div class="p-3 ms-3 bg-danger text-white" style="border-radius: 15px;">
                        <p class="small mb-0">
                            <strong class="text-white"><i class="fas fa-exclamation-triangle me-2"></i>Error:</strong>
                            Sorry, I couldn't get a response. Please try again.
                        </p>
                    </div>
                </div>`;
            $('#chatBox').append(errorMessage);
            $('#chatBox').animate({ scrollTop: $('#chatBox')[0].scrollHeight }, 300);
        }
    });

});

function convertMarkdownToText (markdown) {
    const htmlContent = marked.parse(markdown);
    return separateThinkContent(htmlContent);
}

function separateThinkContent (fullText) {
    const thinkContentRegex = /<think>([\s\S]*?)<\/think>/s; // Capture content inside <think>
    const thinkMatch = fullText.match(thinkContentRegex);

    let thinkParagraph = '';
    let responseContent = fullText;

    if (thinkMatch && thinkMatch[1]) {
        thinkParagraph = thinkMatch[1].trim(); // Captured group is the content inside tags
        responseContent = fullText.replace(thinkContentRegex, '').trim(); // Remove think block from original text
    }
    const uuid = generateUuidV4();
    let button = `<button class="btn btn-outline-primary" type="button" data-bs-toggle="collapse" data-bs-target="#${uuid}" aria-expanded="false" aria-controls="collapseExample">
    Think about this... <spam class="small mb-0">
                            <strong class="text-success"><i class="fas fa-robot me-2"></i>RadioAI:</strong>
                        </spam>
  </button>`;
    let collapseDiv = `<div class="collapse" id="${uuid}">
    <div class="card card-body">
      ${thinkParagraph}
    </div>
  </div>`;

    return button + collapseDiv + responseContent;

}

function generateUuidV4 () {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0,
            v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}



//sweet alert progress alert
function startProgressAlert () {
    Swal.fire({
        title: 'Processing...',
        text: 'Please wait while we process your request.',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });
}

function endProgressAlert () {
    Swal.close();
}