from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import io
from PIL import Image
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import logging
import os

app = Flask(__name__)
CORS(app, resources={r"/predict_pet": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = './stanford_dogs_model.h5'
MAPPING_PATH = './mapping.json'

class CastLayer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        config = super().get_config()
        return config

try:
    model = load_model(MODEL_PATH, custom_objects={'Cast': CastLayer})
    logger.info("Model successfully loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", str(e))
    raise

try:
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    logger.info("Mapping successfully loaded from %s", MAPPING_PATH)
except Exception as e:
    logger.error("Failed to load mapping: %s", str(e))
    raise

# OpenCV ile köpek yüzü tespiti için bir cascade yükleyelim (insan yüzü için, köpekler için çalışmayabilir)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route("/predict_pet", methods=["POST"])
def predict_pet():
    logger.debug("Received request: %s", request.headers)
    logger.debug("Request form: %s", request.form)
    logger.debug("Request files: %s", request.files)

    if 'image' not in request.files:
        logger.error("No image provided in the request")
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        logger.debug("Reading and processing image...")
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_array = np.array(img)
        
        # Model için giriş resmi hazırlığı: 224x224 boyutlandırma ve normalize etme
        resized_img = cv2.resize(img_array, (224, 224))
        resized_img = resized_img.astype("float32") / 255.0
        resized_img = np.expand_dims(resized_img, axis=0)
        
        logger.debug("Making prediction with the model...")
        preds = model.predict(resized_img)
        pred_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][pred_index])
        predicted_label = mapping.get(str(pred_index), "Unknown")
        
        # Köpek yüzü tespiti için OpenCV kullanımı (insan yüzü için tasarlanmış, köpekler için çalışmayabilir)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # İlk tespit edilen yüzü kullan
            (x, y, w, h) = faces[0]
            box = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
            logger.info("Face detected: %s", box)
        else:
            # Yüz bulunamazsa varsayılan kutuyu kullan
            logger.warning("No face detected, using default box")
            box = {
                "x": int(img_array.shape[1] * 0.25),
                "y": int(img_array.shape[0] * 0.25),
                "width": int(img_array.shape[1] * 0.5),
                "height": int(img_array.shape[0] * 0.5)
            }

        logger.info("Prediction successful: label=%s, confidence=%.2f", predicted_label, confidence)
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "detection": {"box": box}
        })

    except Exception as e:
        logger.error("Error processing the image: %s", str(e))
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)