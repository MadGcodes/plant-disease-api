# model_api.py

from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
from flask_cors import CORS # 1. Import CORS

app = Flask(__name__)
CORS(app) # 2. Initialize CORS with your app

# Load your trained YOLOv8 model
model = YOLO('best_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # Read the image file from the request
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Run inference on the image
    # You can adjust conf and other parameters as needed
    results = model.predict(img, conf=0.25, verbose=False)

    # Check if any objects were detected
    if len(results[0].boxes) == 0:
        return jsonify({
            'disease_name': 'No Detection',
            'confidence': 0.0,
            'is_healthy': False # Default to not healthy if nothing is found
        })

    # Extract the top prediction
    top_prediction = results[0].boxes[0] # Get the box with the highest confidence
    disease_class = int(top_prediction.cls)
    disease_name = model.names[disease_class]
    confidence = float(top_prediction.conf)

    # Check if the detected class name contains "healthy"
    is_healthy = 'healthy' in disease_name.lower()

    # Return the result as JSON
    return jsonify({
        'disease_name': disease_name,
        'confidence': confidence,
        'is_healthy': is_healthy
    })

if __name__ == '__main__':
    # Run the app on host 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', port=5000)