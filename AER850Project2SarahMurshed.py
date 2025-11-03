import os
import tensorflow as tf
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  

try:
    import keras
    from keras import layers
    from keras.models import Sequential
    from keras.utils import image_dataset_from_directory
    USING = "keras"
except Exception:
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import image_dataset_from_directory
    USING = "tf.keras"
    keras = tf.keras  

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# STEP 1 - Data Processing
IMG_WIDTH, IMG_HEIGHT = 500, 500  # Resize images to 500x500
BATCH_SIZE = 32  # Batch size  set to 32 for training

# Define the directories where the data is stored
DATA_ROOT =  Path("Data")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "valid"
TEST_DIR  = DATA_ROOT / "test"

OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "my_trained_model.keras"


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Data augmentation for training data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),         # Rescaling images
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
    tf.keras.layers.RandomRotation(0.1),       # Randomly rotate images
    tf.keras.layers.RandomZoom(0.2)            # Randomly zoom images
])

# Only rescaling for validation data
validation_dataset = validation_dataset.map(lambda x, y: (x / 255.0, y))

# Apply data augmentation to the training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))



# STEP 2 - CNN Architecture Design

# Define the model
model = Sequential()

# Convolutional layers with MaxPooling and Dropout
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax'),
])

# STEP 3 - Hyperparameter Analysis

# Compile the model with chosen hyperparameters
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# STEP 4 - Model Evaluation

# Train the model and store the training history
epochs = 20
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)

# Plotting training and validation loss and accuracy
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

model.save(MODEL_PATH)

