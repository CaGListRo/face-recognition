import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import matplotlib.pyplot as plt

def load_and_evaluate_model(model_path: str, test_dir: str, img_size: Tuple[int, int], batch_size: int = 32) -> None:
    """Loads a trained model, evaluates it on the test dataset, and visualizes results.

    Args:
        model_path (str): Path to the trained model file.
        test_dir (str): Directory containing the test dataset.
        img_size (Tuple[int, int]): Target image size (height, width).
        batch_size (int): Batch size for testing. Default is 32.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load the trained model
    model = load_model(model_path)
    print("Model successfully loaded.")

    # Prepare the test data generator
    test_datagen = ImageDataGenerator()
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Make predictions
    predictions = model.predict(test_data)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Visualize predictions
    visualize_predictions(test_data, predictions, predicted_classes)

def visualize_predictions(data_generator, predictions: np.ndarray, predicted_classes: np.ndarray, num_images: int = 8) -> None:
    """Visualizes predictions by showing true and predicted labels for sample images.

    Args:
        data_generator: Image data generator for the test dataset.
        predictions (np.ndarray): Predicted probabilities from the model.
        predicted_classes (np.ndarray): Binary predicted class labels.
        num_images (int): Number of images to visualize. Default is 8.
    """
    class_labels = list(data_generator.class_indices.keys())
    actual_classes = data_generator.classes

    plt.figure(figsize=(15, 15))
    for i in range(min(num_images, len(actual_classes))):
        plt.subplot(3, 3, i + 1)
        image_batch, _ = data_generator[i // data_generator.batch_size]
        image = image_batch[i % data_generator.batch_size]
        true_label = actual_classes[i]
        predicted_label = predicted_classes[i]
        plt.imshow(image.astype("uint8"))
        plt.title(f"True: {class_labels[true_label]}, Pred: {class_labels[predicted_label]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    model_path = "real_fake_face_model.h5"  # Update with your model path
    test_dir = "path_to_test_data"          # Update with your test data path
    img_size = (128, 128)                   # Update based on your model's input size
    batch_size = 8                          # Adjust as needed

    load_and_evaluate_model(model_path, test_dir, img_size, batch_size)
