from flask import Flask, request, jsonify
from predict import predict_single_image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

@app.route('/')
def index():
    return "Number Prediction API"

@app.route('/predict', methods=['POST'])

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    # === Load TFLite model ===
    print("Loading model...")

    try:
        prediction, confidence = predict_single_image(file, interpreter, labels_file_path)  # pass file-like object
        print("this is prediction: ",prediction,"and this is conf: ",type(confidence))
        return jsonify({
        'prediction': prediction,
        'confidence': confidence
        })
    except Exception as e:
        print(f"Error occurred in /predict route: {e}")  # <-- Add this
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)