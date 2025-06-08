import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d, convolve

def compute_xyz_gradients(image_xyz, sigma):
    Ix, Iy, Iz = image_xyz[:, :, 0], image_xyz[:, :, 1], image_xyz[:, :, 2]

    Ix_u = gaussian_filter1d(Ix, sigma=sigma, order=1, axis=1, mode='reflect')
    Ix_v = gaussian_filter1d(Ix, sigma=sigma, order=1, axis=0, mode='reflect')

    Iy_u = gaussian_filter1d(Iy, sigma=sigma, order=1, axis=1, mode='reflect')
    Iy_v = gaussian_filter1d(Iy, sigma=sigma, order=1, axis=0, mode='reflect')

    Iz_u = gaussian_filter1d(Iz, sigma=sigma, order=1, axis=1, mode='reflect')
    Iz_v = gaussian_filter1d(Iz, sigma=sigma, order=1, axis=0, mode='reflect')


    grad_x = np.sqrt(Ix_u**2 + Ix_v**2)
    grad_y = np.sqrt(Iy_u**2 + Iy_v**2)
    grad_z = np.sqrt(Iz_u**2 + Iz_v**2)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return grad_mag


def anisotropic_gaussian_kernel(size, sigma, rho, theta):
    half_size = size // 2
    x, y = np.meshgrid(np.arange(-half_size, half_size+1),
                       np.arange(-half_size, half_size+1))
    coords = np.stack([x, y], axis=-1).astype(np.float32)

    theta_rad = theta
    R_theta = np.array([
        [np.cos(theta_rad), np.sin(theta_rad)],
        [-np.sin(theta_rad), np.cos(theta_rad)]
    ])
    rot_coords = coords @ R_theta.T


    gauss = np.exp(
        -0.5 * ((rot_coords[..., 0] / (sigma * rho)) ** 2 + (rot_coords[..., 1] * rho / sigma) ** 2)
    )
    gauss /= (2 * np.pi * sigma ** 2)


    directional = ((np.cos(theta_rad) * x + np.sin(theta_rad) * y) / (sigma ** 2 / rho ** 2)) * gauss
    return directional

def apply_agdd(image_xyz, sigma, rho, num_directions):
    H, W, _ = image_xyz.shape
    response = np.zeros((H, W), dtype=np.float32)

    for n in range(num_directions):
        theta = n * np.pi / num_directions 
        kernel = anisotropic_gaussian_kernel(size=7, sigma=sigma, rho=rho, theta=theta)

        response_x = convolve(image_xyz[:, :, 0], kernel, mode='reflect')
        response_y = convolve(image_xyz[:, :, 1], kernel, mode='reflect')
        response_z = convolve(image_xyz[:, :, 2], kernel, mode='reflect')

        magnitude = np.sqrt(response_x**2 + response_y**2 + response_z**2)
        response = np.maximum(response, magnitude) 

    return response

def compute_edge_strength_map(grad_xyz, agdd_sigma1, agdd_sigma2):
    esm = (grad_xyz + agdd_sigma1 + agdd_sigma2) / 3.0
    return esm / esm.max()