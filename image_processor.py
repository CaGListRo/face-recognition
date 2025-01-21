import os
from PIL import Image


# load folder names
def get_folder_names(directory) -> list[str]:
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

def process_images(directory) -> None:
    folder_names: list[str] = get_folder_names(directory)
    min_width: int = 999999999
    min_height: int = 999999999
    max_width: int = 0
    max_height: int = 0
    counter: int = 0
    average_width: int = 0
    average_height: int = 0
    for folder in folder_names:
        folder_path: str = os.path.join(directory, folder)
        for file_name in os.listdir(folder_path):
            file_path: str = os.path.join(folder_path, file_name)
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
    print(f"average width= {average_width / counter}, average height= {average_height / counter}")


if __name__ == "__main__":
    directory = "D:/Programmieren/Datasets/LFW/lfw-deepfunneled/lfw-deepfunneled"
    process_images(directory)