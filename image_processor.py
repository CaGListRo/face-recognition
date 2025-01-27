import os
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm

def process_images(directory: str, output_directory: str, image_size: int, normalize_style: int = 0) -> None:
    """Processes images by resizing, normalizing, and saving them to an output directory.

    Args:
        directory (str): Path to the input directory containing subfolders of images.
        output_directory (str): Path to save the processed images.
        image_size (int): Target size for resizing (images will be resized to image_size x image_size).
        normalize_style (int): Normalization style (0 for global normalization, 1 for per-channel normalization).
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Input directory not found: {directory}")

    os.makedirs(output_directory, exist_ok=True)
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    for folder in folder_names:
        folder_path = os.path.join(directory, folder)
        output_folder_path = os.path.join(output_directory, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing folder: {folder}"):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
                try:
                    with Image.open(file_path) as img:
                        img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        img_array = np.array(img_resized, dtype=np.float32)

                        if normalize_style == 1:  # Per-channel normalization
                            for channel in range(img_array.shape[2]):
                                min_val = np.min(img_array[:, :, channel])
                                max_val = np.max(img_array[:, :, channel])
                                img_array[:, :, channel] = (img_array[:, :, channel] - min_val) / (max_val - min_val)
                        else:  # Global normalization
                            min_val = np.min(img_array)
                            max_val = np.max(img_array)
                            img_array = (img_array - min_val) / (max_val - min_val)

                        normalized_img = (img_array * 255).astype(np.uint8)
                        output_path = os.path.join(output_folder_path, file_name)
                        Image.fromarray(normalized_img).save(output_path)

                except Exception as e:
                    print(f"Error processing image {file_name}: {e}")

if __name__ == "__main__":
    # Example usage
    input_directory = "path_to_input_directory"  # Update with your input directory
    output_directory = "path_to_output_directory"  # Update with your output directory
    image_size = 128  # Resize images to 128x128
    normalize_style = 0  # Use global normalization (set to 1 for per-channel normalization)

    process_images(input_directory, output_directory, image_size, normalize_style)