import cv2
import os
import numpy as np
from convert_to_xyz import rgb_to_xyz, save_xyz_image
from cbm3d_filter import cbm3d_denoise_xyz
from esm import compute_xyz_gradients, apply_agdd, compute_edge_strength_map

def process_esm(img_xyz_denoised, path):
    base_name = os.path.splitext(os.path.basename(path))[0]
    result_dir = os.path.join("results", "esm")
    os.makedirs(result_dir, exist_ok=True)

    grad_sigma1 = compute_xyz_gradients(img_xyz_denoised, sigma=np.sqrt(3))
    grad_sigma2 = compute_xyz_gradients(img_xyz_denoised, sigma=np.sqrt(7))

    cv2.imwrite(os.path.join(result_dir, f"{base_name}_gradient_sigma1.png"),
                (grad_sigma1 / grad_sigma1.max() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_gradient_sigma2.png"),
                (grad_sigma2 / grad_sigma2.max() * 255).astype(np.uint8))

    agdd_map_sigma1 = apply_agdd(img_xyz_denoised, sigma=np.sqrt(3), rho=2.0, num_directions=8)
    agdd_map_sigma2 = apply_agdd(img_xyz_denoised, sigma=np.sqrt(7), rho=2.0, num_directions=8)

    cv2.imwrite(os.path.join(result_dir, f"{base_name}_agdd_sigma1.png"),
                (agdd_map_sigma1 / agdd_map_sigma1.max() * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_agdd_sigma2.png"),
                (agdd_map_sigma2 / agdd_map_sigma2.max() * 255).astype(np.uint8))

    esm = compute_edge_strength_map(grad_sigma1, agdd_map_sigma1, agdd_map_sigma2)
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_esm.png"),
                (esm * 255).astype(np.uint8))

    return esm


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

    process_esm(img_xyz_denoised=img_xyz_denoised, path=input_path)


process_image("images/cat.png")
process_image("images/plant.png")
process_image("images/dolphin.png")