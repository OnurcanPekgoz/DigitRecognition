function uploadImage() {
    var formData = new FormData();
    var fileInput = document.getElementById('fileInput');
    var uploadedImage = document.getElementById('uploadedImage');
    var predictionResult = document.getElementById('predictionResult');
    var probabilityResult = document.getElementById('probabilityResult');

    if (fileInput.files.length === 0) {
        alert('Please choose a file first.');
        return;
    }

    formData.append('file', fileInput.files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        uploadedImage.style.display = 'block';
        uploadedImage.src = URL.createObjectURL(fileInput.files[0]);

        predictionResult.innerHTML = 'Prediction: ' + data.prediction;
        probabilityResult.innerHTML = 'Probability: ' + (data.probability * 100).toFixed(2) + '%';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while predicting the digit. Please try again.');
    });
}
