
$(document).ready(function () {
    result_hide();
    $("#result_image_container, #parameter_container").hide();
});



function result_hide () {
    $("#result-show").hide();
    $("#predict-button").hide();
}

function predict_button_show () {
    $("#predict-button").show();
}

let base64Image;

$("#image-selector").change(function () {
    result_hide();
    let reader = new FileReader();
    reader.onload = function (e) {
        let dataURL = reader.result;
        $('#selected-image').attr({ src: dataURL, width: '30%', height: '30%' });
        base64Image = dataURL.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
    };
    reader.readAsDataURL($("#image-selector")[0].files[0]);
    $("#result").text("");
    $("#probability").text("");
    predict_button_show();
});

$("#predict-button").click(function () {
    getPredictionResult();
});

$("#update-button").click(function () {
    getPredictionResult();
});

function getPredictionResult () {
    Swal.fire({
        title: 'Processing...',
        text: 'Please wait while we process your request.',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });

    let message = {
        image: base64Image,
        smooth_samples: 50,
        smooth_noise: 0.1,
        top_labels: 4,
        hide_color: 0,
        num_samples: 1000,
        num_features: 5,
        min_weight: 0.0
    };

    $.post("http://127.0.0.1:5000/predict", JSON.stringify(message),
        function (response) {
            let result = response.prediction.result;
            $("#result").text(result);
            $("#probability").text(response.prediction.accuracy);
            $("#result_image_container").show();
            $("#submit_img").attr({
                'src': "data:image/png;base64," + response.submit_img,
                'width': response.img_size,
                'height': response.img_size
            });

            $("#gradcam_img").attr({
                'src': "data:image/png;base64," + response.gradcam_img,
                'width': response.img_size,
                'height': response.img_size
            });

            $("#lime_explanation").attr({
                'src': "data:image/png;base64," + response.lime_explanation,
                'width': response.img_size,
                'height': response.img_size
            });

            $("#combined_gradcam_lime_exp").attr({
                'src': "data:image/png;base64," + response.combined_gradcam_lime_exp,
                'width': response.img_size,
                'height': response.img_size
            });

            $("#saliency_map").attr({
                'src': "data:image/png;base64," + response.saliency_map,
                'width': response.img_size,
                'height': response.img_size
            });

            $("#lime_explanation_cam").attr({
                'src': "data:image/png;base64," + response.lime_explanation_cam,
                'width': response.img_size,
                'height': response.img_size
            });


            $("#encoded_img").attr('src', "data:image/png;base64," + response.encoded_img);

            Swal.close();
            $("#parameter_container").show();
        })
        .fail(function () {
            Swal.close();
            alert("Error processing the request.");
        });
}
// Update value display when range changes
function updateValue (spanId, value) {
    document.getElementById(spanId).innerText = value;
}
// Enable Bootstrap tooltips
const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

// for RAG app
// Add Document
$('#add_document_btn').click(function () {
    const content = $('#document_content').val();
    if (!content) {
        alert('Please enter document content');
        return;
    }
    $.ajax({
        url: '/add_document',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ content }),
        success: function (response) {
            $('#add_document_status').html(`<div class="alert alert-success">Document added with ID: ${response.id}</div>`);
            $('#document_content').val('');
        },
        error: function () {
            $('#add_document_status').html('<div class="alert alert-danger">Failed to add document</div>');
        }
    });
});

// Query Llama
$('#query_btn').click(function () {
    const query = $('#query_input').val();
    if (!query) {
        alert('Please enter a query');
        return;
    }
    $.ajax({
        url: '/query',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ query }),
        success: function (response) {
            const converter = new showdown.Converter();
            $('#query_results').html(`
                        <h5>Query:</h5>
                        <p>${response.query}</p>
                        <h5>Context:</h5>
                        <p>${response.context}</p>
                        <h5>Response:</h5>
                        <p>${converter.makeHtml(response.response)}</p>
                    `);
        },
        error: function () {
            $('#query_results').html('<div class="alert alert-danger">Failed to retrieve response</div>');
        }
    });
});