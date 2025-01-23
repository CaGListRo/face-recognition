import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(dataset_dir: str, output_dir: str, test_size: float = 0.15, val_size: float = 0.15) -> None:
    """ Split the dataset into training, validation, and test sets. """
    classes = ["real", "fake"]

    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_path)

        # train-test-split
        train_images, test_images = train_test_split(images, test_size = test_size, random_state = 42)
        # train-validation-split
        train_images, val_images = train_test_split(train_images, test_size=val_size / (1 - test_size), random_state = 42)

        # make folder structure
        for split, split_images in tqdm(zip(["train", "val", "test"], [train_images, val_images, test_images])):
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for image in split_images:
                src_path: str = os.path.join(class_path, image)
                dest_path: str = os.path.join(split_dir, image)
                shutil.copy(src_path, dest_path)

if __name__ == "__main__":
    dataset_dir = "D:/Programmieren/Datasets/Real_and_Fake_faces/Processed"
    output_dir = "D:/Programmieren/Datasets/Real_and_Fake_faces/Splitted"
    split_dataset(dataset_dir, output_dir)