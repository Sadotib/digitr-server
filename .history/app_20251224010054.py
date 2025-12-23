from flask import Flask, request, jsonify
from predict import predict_single_image, SimpleCNN, device
import numpy as np
import torch


app = Flask(__name__)

# === Load TFLite model ===
print("Loading model...")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("digitR_cnn.pth", map_location=device))
model.eval()

@app.route('/')
def index():
    return "Number Prediction API"

@app.route('/predict', methods=['POST'])

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    

    try:
        pred = predict_single_image(file, model)
        print(pred)
        return jsonify({
        'prediction': pred,
        })
    except Exception as e:
        print(f"Error occurred in /predict route: {e}")  # <-- Add this
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)