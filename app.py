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

def preprocess_image(image):
    """Görüntüyü model için hazırla"""
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

logger.info("Loading dog breed model and mapping...")
model = load_model(MODEL_PATH, custom_objects={'CastLayer': CastLayer}, compile=False)
logger.info("Model loaded successfully")

with open(MAPPING_PATH, 'r') as f:
    breed_mapping = json.load(f)
logger.info(f"Loaded {len(breed_mapping)} breed mappings")

@app.route('/predict_pet', methods=['POST'])
def predict_pet():
    logger.debug(f"Received request: {dict(request.headers)}")
    logger.debug(f"Request form: {request.form}")
    logger.debug(f"Request files: {request.files}")
    
    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        logger.debug(f"Received file: {file.filename}")
        
        logger.debug("Reading and processing image...")
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        logger.debug("Making prediction with the model...")
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_breeds = []
        
        for idx in top_3_indices:
            breed_name = breed_mapping.get(str(idx), f"Unknown_{idx}")
            confidence = float(predictions[0][idx])
            top_3_breeds.append({
                'breed': breed_name,
                'confidence': confidence
            })
        
        logger.info(f"Prediction successful: {top_3_breeds[0]['breed']} ({top_3_breeds[0]['confidence']:.2%})")
        
        return jsonify({
            'breed': top_3_breeds[0]['breed'],
            'confidence': top_3_breeds[0]['confidence'],
            'top_3': top_3_breeds
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Dog Breed Prediction API',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # Hugging Face default port
    app.run(host='0.0.0.0', port=port, debug=False)
