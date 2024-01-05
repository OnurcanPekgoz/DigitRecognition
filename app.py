from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile

app = Flask(__name__)

model = load_model('mnist_cnn_model.h5')


def predict_digit(file):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        file.save(temp_path)

        img = image.load_img(temp_path, target_size=(28, 28), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        probability = prediction[0, predicted_digit]

    os.remove(temp_path)

    return predicted_digit, probability

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        predicted_digit, probability = predict_digit(file)
        return jsonify({'prediction': int(predicted_digit), 'probability': float(probability)})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
