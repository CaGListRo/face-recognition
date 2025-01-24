import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from PIL import Image
import matplotlib as plt
from typing import Final


# file paths
train_dir: str = "D:/Programmieren/Datasets/Real_and_Fake_faces/Splitted/train"
test_dir: str = "D:/Programmieren/Datasets/Real_and_Fake_faces/Splitted/test"
val_dir: str = "D:/Programmieren/Datasets/Real_and_Fake_faces/Splitted/val"

# image size must fit to the images
IMG_SIZE: Final[tuple[int]] = (128, 128)
BATCH_SIZE: Final[int] = 8

# prepare data for the ImageDataGenerator
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# load images from the paths
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary'
)

# define model
model = Sequential([
    Conv2D(32, (3, 3), activation = "relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(1, activation = "sigmoid")  # binary classification (real vs. fake)
])

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

# train model
history = model.fit(
    train_data,
    validation_data = val_data,
    epochs = 10
)

# save model
model.save("real_fake_face_model.h5")
print("Model saved successfully.")

# test model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")