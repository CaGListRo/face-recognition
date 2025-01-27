from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple

def create_cnn_model(input_shape: Tuple[int, int, int], dropout_rate: float = 0.5) -> Sequential:
    """Creates a Convolutional Neural Network (CNN) model for binary classification.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input images (height, width, channels).
        dropout_rate (float): Dropout rate to prevent overfitting. Default is 0.5.

    Returns:
        Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir: str, val_dir: str, img_size: Tuple[int, int], batch_size: int, epochs: int, model_save_path: str) -> None:
    """Trains a CNN model on the given dataset and saves it to a file.

    Args:
        train_dir (str): Directory containing the training data.
        val_dir (str): Directory containing the validation data.
        img_size (Tuple[int, int]): Target size for the input images (height, width).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        model_save_path (str): Path to save the trained model.
    """
    # Prepare data generators
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # Create model
    input_shape = (img_size[0], img_size[1], 3)  # Assuming RGB images
    model = create_cnn_model(input_shape)

    # Train the model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Save the model
    model.save(model_save_path)
    print(f"Model successfully saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    train_dir = "path_to_train_directory"        # Update with the actual path
    val_dir = "path_to_val_directory"            # Update with the actual path
    img_size = (128, 128)                        # Resize images to 128x128
    batch_size = 32                              # Adjust batch size as needed
    epochs = 10                                  # Number of training epochs
    model_save_path = "real_fake_face_model.h5"  # Path to save the trained model

    train_model(train_dir, val_dir, img_size, batch_size, epochs, model_save_path)
