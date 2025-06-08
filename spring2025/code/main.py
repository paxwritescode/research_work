import cv2
import sys
print(sys.path)
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_to_xyz import rgb_to_xyz, save_xyz_image
from cbm3d_filter import cbm3d_denoise_xyz

def process_image(input_path):
    img_rgb = cv2.imread(input_path)
    if img_rgb is None:
        raise FileNotFoundError(f"Failed to load image at path:: {input_path}")
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_xyz = rgb_to_xyz(img_rgb)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    save_xyz_image(img_xyz, filename=f"{base_name}_xyz", output_dir="results/convert_to_xyz")

    img_xyz_denoised = cbm3d_denoise_xyz(img_xyz)
    save_xyz_image(img_xyz_denoised, filename=f"{base_name}_xyz_cbm3d", output_dir="results/cbm3d")

process_image("images/cat.png")
process_image("images/plant.png")
process_image("images/dolphin.png")