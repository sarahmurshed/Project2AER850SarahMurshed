# STEP 5 - Model Testing
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



MODEL_PATH = Path("outputs") / "my_trained_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Importing required packages for model testing
from keras.preprocessing import image

# Define image size
IMG_WIDTH, IMG_HEIGHT = 500, 500

# Class labels (in the same order as they were used during training)
class_labels = ['Crack', 'Missing-Head', 'Paint-Off']

# Function to load, preprocess, and predict the class of an image
def process_and_predict(img_path):
    # Load the image and resize it to match the input shape of the model
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    # Convert the image to an array and normalize it
    img_array = image.img_to_array(img) / 255.0
    # Add a batch dimension to the image
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class of the image
    prediction = model.predict(img_array)
    # Find the index of the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Confidence level

    return img, predicted_class, confidence
# Test images paths
DATA_ROOT = Path("Data")

test_images = [
    (DATA_ROOT / "test" / "crack" / "test_crack.jpg", "Crack"),
    (DATA_ROOT / "test" / "missing-head" / "test_missinghead.jpg", "Missing-Head"),
    (DATA_ROOT / "test" / "paint-off" / "test_paintoff.jpg", "Paint-Off")
]


# Plotting the test images with actual and predicted labels
fig, axes = plt.subplots(1, len(test_images), figsize=(15, 5))
for i, (img_path, true_label) in enumerate(test_images):
    # Process and predict each test image
    img, predicted_class, confidence = process_and_predict(img_path)

    # Display the image
    axes[i].imshow(img)
    axes[i].axis('off')

    # Set the title with the actual label, predicted label, and confidence
    predicted_label = class_labels[predicted_class]
    axes[i].set_title(f"Actual: {true_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}%")
