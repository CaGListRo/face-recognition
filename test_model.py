import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. load model
model_path = "real_fake_face_model.h5"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# 2. prepare test data
test_dir = "D:/Programmieren/Datasets/Real_and_Fake_faces/Splitted/test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

test_datagen = ImageDataGenerator()
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  # Images must stay in the same order to fit with their labels
)

# 3. evaluate model
test_loss, test_acc = model.evaluate(test_data)
print(f"Test loss: {test_loss:.2f}")
print(f"Test accuracy: {test_acc:.2f}")

# 4. generate prediction
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Binary threshold at 0.5

# 5. visualize outcome
def visualize_predictions(data_generator, predictions, predicted_classes, num_images=8):
    class_labels = list(data_generator.class_indices.keys())  # class labels (z. B. ['fake', 'real'])
    actual_classes = data_generator.classes

    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        image, label = data_generator[i // data_generator.batch_size][0][i % data_generator.batch_size], actual_classes[i]
        predicted_label = predicted_classes[i]
        plt.imshow(image.astype("uint8"))
        plt.title(f"True: {class_labels[label]}, Pred: {class_labels[predicted_label]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# visualize predictions for some test images
visualize_predictions(test_data, predictions, predicted_classes, num_images=8)