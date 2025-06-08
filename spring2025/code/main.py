import cv2
import os
from convert_to_xyz import rgb_to_xyz, show_channels, save_xyz_image

def process_image(input_path):
    img_rgb = cv2.imread(input_path)
    if img_rgb is None:
        raise FileNotFoundError(f"Failed to load image at path:: {input_path}")
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_xyz = rgb_to_xyz(img_rgb)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    save_xyz_image(img_xyz, filename=f"{base_name}_xyz", output_dir="results/convert_to_xyz")

process_image("images/cat.png")