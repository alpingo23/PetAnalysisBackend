---
title: Dog Breed Prediction API
emoji: 🐕
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Dog Breed Prediction API

This is a Flask API for predicting dog breeds using a TensorFlow deep learning model trained on the Stanford Dogs dataset.

## Features

- 🐕 Predicts dog breeds from images
- 🎯 Returns top 3 most likely breeds with confidence scores
- 🚀 Fast inference with TensorFlow
- 🌐 CORS enabled for web applications

## API Endpoints

### POST /predict_pet
Upload an image to get dog breed predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` file

**Response:**
```json
{
  "breed": "Golden_Retriever",
  "confidence": 0.95,
  "top_3": [
    {"breed": "Golden_Retriever", "confidence": 0.95},
    {"breed": "Labrador_Retriever", "confidence": 0.03},
    {"breed": "Irish_Setter", "confidence": 0.01}
  ]
}
```

### GET /
Health check endpoint.

## Model

- Based on Stanford Dogs Dataset
- 120 dog breeds supported
- Input: 224x224 RGB images
- Model: Custom TensorFlow/Keras CNN

## Usage

```bash
curl -X POST -F "image=@dog.jpg" https://YOUR_SPACE_URL/predict_pet
```

## License

MIT
