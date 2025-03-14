from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Etiketleri yükleyin (örneğin, dataset/labels.csv)
labels_df = pd.read_csv('dataset/labels.csv')
class_names = labels_df["breed"].tolist()

# Modeli yükleyin (model/dogclassification.h5)
model = tf.keras.models.load_model("model/dogclassification.h5")

def predict_breed(img_path):
    # Modelin eğitiminde kullanılan boyutu kontrol edin, örneğin 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    return class_names[predicted_class]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya gönderilmedi.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya adı boş.'}), 400

    # Dosyayı geçici kaydedin
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filepath = os.path.join(temp_dir, file.filename)
    file.save(filepath)

    try:
        breed = predict_breed(filepath)
        return jsonify({'predicted_breed': breed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
