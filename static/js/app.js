
$(document).ready(function () {
    result_hide();
    load_bg();
    $("#result_image_container").hide();
});


$("#smooth_samples, #smooth_noise, #top_labels, #hide_color, #num_samples").on('input', function () {
    $("#update-button").prop('disabled', false);
});
function load_bg () {
    $("#main-bg").removeClass();
}

function result_hide () {
    $("#result-show").hide();
    $("#predict-button").hide();
}

function predict_button_show () {
    $("#predict-button").show();
}

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
        top_labels: 5,
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
            $("#gradcam_img").attr('src', "data:image/png;base64," + response.gradcam_img);
            $("#lime_explanation").attr('src', "data:image/png;base64," + response.lime_explanation);
            $("#combined_gradcam_lime_exp").attr('src', "data:image/png;base64," + response.combined_gradcam_lime_exp);
            $("#saliency_map").attr('src', "data:image/png;base64," + response.saliency_map);

            Swal.close();
        })
        .fail(function () {
            Swal.close();
            alert("Error processing the request.");
        });
});

$("#update-button").click(function () {
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
        smooth_samples: parseInt($("#smooth_samples").val()),
        smooth_noise: parseFloat($("#smooth_noise").val()),
        top_labels: parseInt($("#top_labels").val()),
        hide_color: parseInt($("#hide_color").val()),
        num_samples: parseInt($("#num_samples").val()),
        num_features: parseInt($("#num_features").val()),
        min_weight: parseFloat($("#min_weight").val()),
    };

    $.post("http://127.0.0.1:5000/predict", JSON.stringify(message),
        function (response) {
            let result = response.prediction.result;
            $("#result").text(result);
            $("#probability").text(response.prediction.accuracy);
            if (result === "Covid19 Negative") {
                normal_warning_bg();
            }
            if (result === "Covid19 Positive") {
                covid_warning_bg();
            }
            $("#result_image_container").show();
            $("#gradcam_img").attr('src', "data:image/png;base64," + response.gradcam_img);
            $("#lime_explanation").attr('src', "data:image/png;base64," + response.lime_explanation);
            $("#combined_gradcam_lime_exp").attr('src', "data:image/png;base64," + response.combined_gradcam_lime_exp);
            $("#saliency_map").attr('src', "data:image/png;base64," + response.saliency_map);

            Swal.close();
        })
        .fail(function () {
            Swal.close();
            alert("Error processing the request.");
        });
});

function normal_warning_bg () {
    $("#main-bg").removeClass("bg-danger");
    $("#main-bg").addClass("bg-success");
}

function load_bg () {
    $("#main-bg").removeClass();
}

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
        // $('#selected-image').attr("src", dataURL);
        $('#selected-image').attr({ src: dataURL, width: '30%', height: '30%' });
        base64Image = dataURL.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");

    };
    reader.readAsDataURL($("#image-selector")[0].files[0]);
    $("#result").text("");
    $("#probability").text("");
    predict_button_show();
});

$("#predict-button").click(function () {
    // Show the SweetAlert2 loading spinner
    Swal.fire({
        title: 'Processing...',
        text: 'Please wait while we process your request.',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });

    let message = {
        image: base64Image
    };

    $.post("http://127.0.0.1:5000/predict", JSON.stringify(message),
        function (response) {
            let result = response.prediction.result;
            $("#result").text(result);
            $("#probability").text(response.prediction.accuracy);
            if (result === "Covid19 Negative") {
                normal_warning_bg();
            }
            if (result === "Covid19 Positive") {
                covid_warning_bg();
            }
            console.log(response);
            $("#result_image_container").show();
            img = "data:image/png;base64," + response.visualization;

            $("#result_image").attr('src', img);

            // Close the SweetAlert2 spinner
            Swal.close();
        })
        .fail(function () {
            // Close the SweetAlert2 spinner on error
            Swal.close();
            alert("Error processing the request.");
        });
});
