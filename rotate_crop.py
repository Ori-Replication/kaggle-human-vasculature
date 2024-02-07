import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import rotate
from tqdm import tqdm

def load_images(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tif"):
            image_path = os.path.join(folder_path, filename)
            img = io.imread(image_path, as_gray=True)
            images.append((filename, img))
    return images

def rotate_and_crop_volume(volume, axis1, angle1, axis2, angle2, crop_size):
    rotated_images = []

    for filename, layer in tqdm(volume, desc="Processing", unit="image"):
        rotated_layer = rotate(layer, angle1, axes=(axis1, 1-axis1), reshape=False, mode='constant', cval=0.0)
        rotated_layer = rotate(rotated_layer, angle2, axes=(axis2, 1-axis2), reshape=False, mode='constant', cval=0.0)

        center_x, center_y = np.array(rotated_layer.shape) // 2
        cropped_layer = rotated_layer[center_x - crop_size//2:center_x + crop_size//2, 
                                      center_y - crop_size//2:center_y + crop_size//2]

        rotated_images.append((filename, cropped_layer))

    return rotated_images

def visualize_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def save_images(images, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename, img in images:
        output_path = os.path.join(output_folder, filename)
        io.imsave(output_path, img)

def process_folders(input_root, output_root, crop_size, rotation_angle_x, rotation_angle_y):
    for folder_name in ["kidney_1_dense", "kidney_1_voi", "kidney_3_dense"]:
        input_folder = os.path.join(input_root, folder_name)
        output_folder_images = os.path.join(output_root, f"{folder_name}_rotated_cropped/images")
        output_folder_labels = os.path.join(output_root, f"{folder_name}_rotated_cropped/labels")

        # Load and process images
        ct_volume_images = load_images(os.path.join(input_folder, "images"))
        rotated_volume_images = rotate_and_crop_volume(ct_volume_images, 0, rotation_angle_x, 1, rotation_angle_y, crop_size)
        save_images(rotated_volume_images, output_folder_images)

        # Load and process labels (if they exist)
        ct_volume_labels = load_images(os.path.join(input_folder, "labels"))
        rotated_volume_labels = rotate_and_crop_volume(ct_volume_labels, 0, rotation_angle_x, 1, rotation_angle_y, crop_size)
        save_images(rotated_volume_labels, output_folder_labels)

if __name__ == "__main__":
    input_root = r'D:/dataset/train'
    output_root = r'D:/dataset/train_output' # save to a new folder
    crop_size = 512
    rotation_angle_x = 15
    rotation_angle_y = 15

    process_folders(input_root, output_root, crop_size, rotation_angle_x, rotation_angle_y)
