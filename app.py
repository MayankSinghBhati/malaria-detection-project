from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import time
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("\n==============================")
print("🦠 MALARIA DETECTION STARTING")
print("==============================")

# ======================
# Load Model (H5 Only)
# ======================
MODEL_PATH = "best_model.h5"
CLASS_NAMES = ['Parasitized', 'Uninfected']

model = None

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully")
else:
    print("✗ Model file not found")

# ======================
# Image Preprocessing
# ======================
def prepare_image(image_path):
    try:
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((96, 96))
        img_array = np.array(img, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print("Image processing error:", e)
        return None

# ======================
# Routes
# ======================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # Save uploaded image
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, file.filename)
    file.save(image_path)

    # Prepare image
    img_array = prepare_image(image_path)

    if img_array is None:
        return jsonify({'success': False, 'error': 'Image processing failed'}), 500

    # Prediction
    start_time = time.time()
    prediction = model.predict(img_array, verbose=0)
    end_time = time.time()

    prob = float(prediction[0][0])
    processing_time = round(end_time - start_time, 2)

    if prob > 0.5:
        result = "Uninfected"
        confidence = prob * 100
        parasitized_prob = (1 - prob) * 100
        uninfected_prob = prob * 100
    else:
        result = "Parasitized"
        confidence = (1 - prob) * 100
        parasitized_prob = (1 - prob) * 100
        uninfected_prob = prob * 100

    return render_template(
        'result.html',
        prediction=result,
        confidence=round(confidence, 2),
        parasitized_prob=round(parasitized_prob, 2),
        uninfected_prob=round(uninfected_prob, 2),
        processing_time=processing_time,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        image_file=file.filename,
        model_name="Custom CNN"
    )


@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })


# Render production run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)