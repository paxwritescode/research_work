import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def SUAMM(E_d, cH, cV, cD):
    """
    Parameters:
    ------------
    E_d : ndarray
        Low-frequency feature map after MSANM
    cH, cV, cD : ndarray
         High-frequency component of the original image

    Returns:
    --------
    CH_prime, CV_prime, CD_prime : ndarray
        Filtered components after SUAMM
    """

    Cx = sobel(E_d, axis=1)  # горизонтальное направление (∂/∂x)
    Cy = sobel(E_d, axis=0)

    delta = np.sqrt(Cx**2 + Cy**2)
    theta = np.arctan2(Cy, Cx)
    theta = np.mod(theta, 2 * np.pi)

    delta = np.sqrt(cH**2 + cV**2 + cD**2)

    theta_H = ((theta >= 0) & (theta < np.pi/8)) | ((theta >= 15*np.pi/8) & (theta < 2*np.pi)) | \
              ((theta >= 7*np.pi/8) & (theta < 9*np.pi/8))

    theta_V = ((theta >= 3*np.pi/8) & (theta < 5*np.pi/8)) | ((theta >= 11*np.pi/8) & (theta < 13*np.pi/8))

    theta_D = ((theta >= np.pi/8) & (theta < 3*np.pi/8)) | ((theta >= 9*np.pi/8) & (theta < 11*np.pi/8)) | \
              ((theta >= 5*np.pi/8) & (theta < 7*np.pi/8)) | ((theta >= 13*np.pi/8) & (theta < 15*np.pi/8))

    T_delta = 0.5 * (np.max(delta) + np.min(delta))
    
    mask = ((theta_H | theta_V | theta_D) & (delta > T_delta))

    CH_prime = np.where(mask, cH, 0)
    CV_prime = np.where(mask, cV, 0)
    CD_prime = np.where(mask, cD, 0)

    return CH_prime, CV_prime, CD_prime

def visualize_SUAMM(CH_prime, CV_prime, CD_prime, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(CH_prime, cmap='gray')
    axes[0].set_title('CH\' после SUAMM')
    axes[0].axis('off')

    axes[1].imshow(CV_prime, cmap='gray')
    axes[1].set_title('CV\' после SUAMM')
    axes[1].axis('off')

    axes[2].imshow(CD_prime, cmap='gray')
    axes[2].set_title('CD\' после SUAMM')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
