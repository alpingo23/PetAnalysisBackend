# Dog Breed Identifier

## Overview
The Dog Breed Identifier is a Python program that utilizes a pre-trained ResNet50V2 model to classify images of dogs into their respective breeds. The model is fine-tuned and trained on a dataset of dog images labeled with their breeds, which are stored in the `train` folder and described in the `labels.csv` file.

## How It Works
- **Data Preparation:**
  - The program reads breed labels from `labels.csv`.
  - Images from the `train` folder are resized and preprocessed.
  - The dataset is split into training and testing sets.
- **Data Augmentation:**
  - Image data augmentation is applied to the training set to enhance the model's performance.
- **Model Training:**
  - A ResNet50V2 model pre-trained on ImageNet is used.
  - The top layers of the model are customized for dog breed classification.
  - Training occurs over 20 epochs with callbacks for reducing learning rate and early stopping.
![image](https://github.com/tawsifrm/Dog-Breed-Identifier-OpenCV/assets/121325051/e6457378-e79d-4b11-8824-6dade7ddec1c)

- **Prediction:**
  - An image of a dog (e.g., 'germanshepherd.jpg') can be classified into its breed using this trained model.

## Files Included
Below are the files included in this project:
- `dog-breed-identification.py`: The main Python script that contains the code for data preparation, model training, and prediction.
- `labels.csv`: A CSV file that contains the mapping of image IDs to breed names.
- `model`: A folder that contains the saved model file (`model`) and the weights file (`model.weights`).
- `test`: A folder that contains three test images of different dog breeds.
- `train`: A folder that contains the training images of 60 dog breeds.

## Running the Program
Ensure you have all required libraries installed. 
The following Python libraries are used in this project. Make sure to use the latest version of each for compatibility:
- TensorFlow (2.2 or higher)
- Keras (if TensorFlow version is below 2.2, install Keras separately)
- OpenCV
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

You can run the program using:
```bash
python dog-breed-identification.py
```

This will train the model and predict the breed of a specified dog image. You can change the image path in the code to test different images. The output will print the predicted breed name for the image. For example:
```python
Predicted Breed for this Dog is : ['leonberg']
```

![image](https://github.com/tawsifrm/Dog-Breed-Identifier-OpenCV/assets/121325051/7d4f25e2-d0b6-4f9b-8f94-d3f87da9d403)

