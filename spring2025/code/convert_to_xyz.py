import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def rgb_to_xyz(image_rgb):

    # normalization
    img_rgb_norm = image_rgb / 255.0

    # transformation matrix
    T_rgb2xyz = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]    
    ])

    # transformation RGB -> XYZ
    shape = img_rgb_norm.shape
    img_flat = img_rgb_norm.reshape(-1, 3)
    img_xyz_flat = img_flat @ T_rgb2xyz.T
    img_xyz = img_xyz_flat.reshape(shape)

    return img_xyz


def save_xyz_image(image_xyz, filename, output_dir = 'results/convert_to_xyz'):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filename))[0]

    channels = []
    labels = ['X', 'Y', 'Z']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (255,)

    for i in range(3):
        channel = image_xyz[:, :, i]
        channel_normalized = channel / np.max(channel)
        channel_8bit = np.clip(channel_normalized * 255.0, 0, 255).astype(np.uint8)

        channel_bgr = cv2.cvtColor(channel_8bit, cv2.COLOR_GRAY2BGR)
        cv2.putText(channel_bgr, f'Channel {labels[i]}', (10, 30), font, font_scale, (0, 0, 0), thickness)
        
        channels.append(channel_bgr)
    combined = np.hstack(channels)

    output_path = os.path.join(output_dir, f'{base_name}.png')
    cv2.imwrite(output_path, combined)
    print(f"Combined image saved: {output_path}")