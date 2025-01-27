import os
import shutil
from sklearn.model_selection import train_test_split
from typing import List

def split_dataset(dataset_dir: str, output_dir: str, test_size: float = 0.15, val_size: float = 0.15) -> None:
    """Splits a dataset into training, validation, and test sets.

    Args:
        dataset_dir (str): Path to the dataset directory containing subfolders for each class.
        output_dir (str): Directory where the split dataset will be saved.
        test_size (float): Proportion of the dataset to use for testing.
        val_size (float): Proportion of the remaining training dataset to use for validation.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Split data into train, test, and validation
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        train_images, val_images = train_test_split(train_images, test_size=val_size / (1 - test_size), random_state=42)

        for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for image in split_images:
                src = os.path.join(class_path, image)
                dst = os.path.join(split_dir, image)
                shutil.copy(src, dst)

if __name__ == "__main__":
    # Example usage
    dataset_dir = "path_to_dataset"  # Update with the actual dataset path
    output_dir = "path_to_output"    # Update with the desired output path
    split_dataset(dataset_dir, output_dir)
