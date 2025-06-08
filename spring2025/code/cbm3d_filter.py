import numpy as np
from bm3d import bm3d
import cv2

def xyz_to_yuv(image_xyz):
    #XYZ2YUV matrix
    T = np.array([
        [0.3, 0.59, 0.11],
        [-0.17, -0.33, 0.5],
        [0.5, -0.42, -0.08]
    ])
    shape = image_xyz.shape
    flat_xyz = image_xyz.reshape(-1, 3)
    yuv_flat = flat_xyz @ T.T
    image_yuv = yuv_flat.reshape(shape)
    return image_yuv

def yuv_to_xyz(image_yuv):
    T = np.array([
        [0.3, 0.59, 0.11],
        [-0.17, -0.33, 0.5],
        [0.5, -0.42, -0.08]
    ])
    T_inv = np.linalg.inv(T)
    shape = image_yuv.shape
    flat_yuv = image_yuv.reshape(-1, 3)
    xyz_flat = flat_yuv @ T_inv.T
    image_xyz = xyz_flat.reshape(shape)
    return image_xyz

def cbm3d_denoise_xyz(image_xyz, sigma=0.05):
    image_yuv = xyz_to_yuv(image_xyz)

    denoised_channels = []
    for i in range(3):
        channel = image_yuv[:, :, i]
        print(f"Channel {i}: min={channel.min()}, max={channel.max()}")
        channel_norm = (channel -  channel.min())/(channel.max() - channel.min() + 1e-8)
        denoised = bm3d(channel_norm, sigma_psd=sigma)
        denoised = denoised * (channel.max() - channel.min()) + channel.min()
        denoised_channels.append(denoised)

    denoised_yuv = np.stack(denoised_channels, axis=2)
    denoised_xyz = yuv_to_xyz(denoised_yuv)

    return denoised_xyz