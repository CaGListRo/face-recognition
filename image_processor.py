import os
import numpy as np
from tqdm import tqdm
from PIL import Image


# load folder names
def get_folder_names(directory: str) -> list[str]:
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

def process_images(directory: str, normalize_style: int, save_directory) -> None:
    folder_names: list[str] = get_folder_names(directory)
    # define min and max values
    min_width: int = 999999999
    min_height: int = 999999999
    max_width: int = 0
    max_height: int = 0
    # initialize the counter
    counter: int = 0
    # define average values
    average_width: int = 0
    average_height: int = 0

    # iterate over the folders to check the image dimensions
    # for folder in tqdm(folder_names):
    #     folder_path: str = os.path.join(directory, folder)

    for file_name in os.listdir(directory):
        file_path: str = os.path.join(directory, file_name)

        if file_name.lower().endswith((".jpg", ".png", ".tiff", ".jpeg", ".bmp")):
            try:
                with Image.open(file_path) as img:
                    counter += 1
                    width, height = img.size

                    average_width += width
                    average_height += height

                    if width < min_width:
                        min_width = width
                    if height < min_height:
                        min_height = height
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height
                        
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")

    print(f"min width= {min_width}, min height= {min_height}")
    print(f"max width= {max_width}, max height= {max_height}")
    if counter > 0:
        print(f"average width= {average_width / counter}, average height= {average_height / counter}")

    target_size: tuple[int] = (min(128, min_width), min(128, min_height))

    # if min_width == max_width and min_height == max_height:
        # if not os.path.exists(directory + "/processed_images"):
        #     os.mkdir(directory + "/processed_images")
    # for folder in tqdm(folder_names):
    #     folder_path: str = os.path.join(directory, folder)

    for file_name in os.listdir(directory):
        file_path: str = os.path.join(directory, file_name)
        if file_name.lower().endswith((".jpg", ".png", ".tiff", ".jpeg", ".bmp")):
            try:
                with Image.open(file_path) as image:
                    # resize image
                    image: Image = image.resize(target_size, Image.Resampling.LANCZOS)
                    # normalize image
                    image_array: np.array = np.array(image, dtype=np.float32)

                    if normalize_style == 1:
                        for channel in range(image_array.shape[2]):
                            min_val = np.min(image_array[:, :, channel])
                            max_val = np.max(image_array[:, :, channel])
                            image_array[:, :, channel] = (image_array[:, :, channel] - min_val) / (max_val - min_val)

                    else:
                        min_val = np.min(image_array)
                        max_val = np.max(image_array)
                        image_array = (image_array - min_val) / (max_val - min_val)

                    # save image
                    normalized_image_uint8 = (image_array * 255).astype(np.uint8)
                    normalized_image_pil = Image.fromarray(normalized_image_uint8)
                    normalized_image_pil.save(save_directory + file_name)
                
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")

    # else:
    #     print("Images must have the same width and the same height.")

    print("Done")


if __name__ == "__main__":
    directory: str = "D:/Programmieren/Datasets/Real_and_Fake_faces/real_and_fake_face/training_fake/"
    save_directory: str = "D:/Programmieren/Datasets/Real_and_Fake_faces/Processed/fake/"  # put one / at the end
    normalize_style: int = 0  # 0 = Normalize over all channels, 1 = Normalize over each channel separate
    process_images(directory, normalize_style, save_directory)