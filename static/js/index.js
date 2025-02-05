$(document).ready(function () {

    const probabilitiesDiv = $('#probabilities');
    let imgWidth;
    let imgHeight;

    const modal = new bootstrap.Modal(document.getElementById('uploadModal'), {
        backdrop: 'static', // Prevents closing when clicking outside
        keyboard: false // Prevents closing with the ESC key
    });

    $('#uploadForm').on('submit', function (e) {
        e.preventDefault();

        modal.show();

        // Show the progress spinner
        $('#progressSpinner').show();
        // disable submit button
        $('#submitModalForm').prop('disabled', true);

        // Prepare the form data for file upload
        const formData = new FormData();
        const fileInput = $('#imageUpload')[0];
        formData.append('file', fileInput.files[0]);

        // Perform the AJAX request to upload the file
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                // Hide the progress spinner and show the form
                $('#progressSpinner').hide();
                // active submit button
                $('#submitModalForm').prop('disabled', false);

                // Extract the file and prediction from the response
                const file = fileInput.files[0];
                const prediction = response.prediction;

                // Process the uploaded image and display results
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function (event) {
                        const uploadedImageSrc = event.target.result;
                        const img = new Image();
                        img.src = event.target.result;
                        img.onload = function () {
                            imgWidth = img.naturalWidth;
                            imgHeight = img.naturalHeight;

                            // Update the UI with the uploaded image and prediction results
                            $('#uploadedImage').attr('src', uploadedImageSrc).css({ "max-width": "100%" });
                            $('#predictionResult').text(`Prediction: ${prediction}`);
                            probabilitiesDiv.empty();
                            for (const [label, prob] of Object.entries(response.probabilities)) {
                                probabilitiesDiv.append(`  ${label}: ${Math.round(prob * 100)}%,  `);
                            }
                            $('#grad-cam').attr('src', 'data:image/png;base64,' + response.grad_cam).css({ "max-width": "100%" });
                            $('#lime').attr('src', 'data:image/png;base64,' + response.lime).css({ "max-width": "100%" });
                            // $('#grayscale_viz').attr('src', 'data:image/png;base64,' + response.grayscale_viz).css({ "max-width": "100%" });
                            // $('#overlay_viz').attr('src', 'data:image/png;base64,' + response.overlay_viz).css({ "max-width": "100%" });

                            $('#results').show();
                        };
                    };
                    reader.readAsDataURL(file); // Read the file as a data URL
                }
            },
            error: function (error) {
                console.error('Error:', error);
                modal.hide(); // Hide the modal in case of an error
            }
        });
    });

    $('#submitModalForm').off('click').on('click', function () {
        $(".image-container").css({ "width": imgWidth, "height": imgHeight });
        $("#uploadedImage, #grad-cam, #lime, #grayscale_viz, #overlay_viz").css({ "width": imgWidth * 0.75, "height": imgHeight * 0.75 });
        $('.image-container').resizable();
        startProgressAlert();
        modal.hide();
        const medical_history = $('#medical_history').val();
        const pt_description = $('#pt_description').val();
        const symptoms = $('#symptoms').val();
        const prediction = $('#probabilities').text();
        const fileInput = $('#imageUpload')[0];

        const formData = {
            medical_history: medical_history,
            pt_description: pt_description,
            symptoms: symptoms,
            prediction: prediction,
            message: 'this is the initial message so make sure that analysis all data and give proper plan',  // Assuming you want to pass this too
        };

        if (fileInput.files[0]) {
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onloadend = function () {
                formData.file = reader.result;  // This will be base64 encoded
                sendData(formData);
            };

            reader.readAsDataURL(file);  // Convert the file to base64
        } else {
            sendData(formData);
        }

        function sendData (formData) {
            $.ajax({
                url: '/chat',
                type: 'POST',
                contentType: 'application/json',  // Set to JSON content type
                data: JSON.stringify(formData),  // Send data as a JSON string
                success: function (response) {
                    endProgressAlert();
                    $('#chatBox').append(`<p><strong>RadioAI:</strong> ${convertMarkdownToText(response.response)}</p>`);
                    console.log('Additional data saved successfully.', response);

                },
                error: function (xhr) {
                    console.error('Error saving additional data:', xhr.responseText);
                }
            });
        }
    });


    $('#sendChat').on('click', function () {
        const message = $('#chatInput').val();
        if (message.trim() === '') return;

        $('#chatBox').append(`<p><strong>You:</strong> ${message}</p>`);
        $('#chatInput').val('');
        Swal.fire({
            title: 'Processing...',
            text: 'Please wait while we process your request.',
            allowOutsideClick: false,
            didOpen: () => {
                Swal.showLoading();
            }
        });
        const formData = {
            message: message,
        };
        $.ajax({
            url: '/chat',
            type: 'POST',
            contentType: 'application/json',  // Set to JSON content type
            data: JSON.stringify(formData),  // Send data as a JSON string
            success: function (response) {
                endProgressAlert();
                $('#chatBox').append(`<p><strong>RadioAI:</strong> ${convertMarkdownToText(response.response)}</p>`);
                console.log('Additional data saved successfully.', response);

            },
            error: function (xhr) {
                console.error('Error saving additional data:', xhr.responseText);
            }
        });

        Swal.close();
    });

    function convertMarkdownToText (markdown) {
        const htmlContent = marked.parse(markdown);
        return htmlContent;
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

    $('textarea').on('input', function () {
        $(this).height('auto');  // Reset the height to auto
        $(this).height(this.scrollHeight);  // Set the height to fit content
    });
    // Initialize tooltips
    var tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    $(window).on('resize', function () {
        var parentHeight = $('#chatBox').parent().height(); // Get the height of the parent container
        $('#chatBox').height(parentHeight);  // Set the height of chatBox dynamically
    });

});
