import os
from PIL import Image


def resize_images_in_folder(folder_path, size):
    """
    Resizes all images in the specified folder and its subfolders to the given size.

    :param folder_path: Path to the folder containing images.
    :param size: Tuple indicating the desired size (width, height).
    """
    # Iterate through the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                file_path = os.path.join(root, file)

                # Open the image file
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize(size)

                    # Save the resized image back to the same path
                    resized_img.save(file_path)
                print(f"Resized image: {file_path}")


folder_path = "../result"
size = (512, 512)

resize_images_in_folder(folder_path, size)
