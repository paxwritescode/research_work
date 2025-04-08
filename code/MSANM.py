import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.util import img_as_float
import matplotlib.pyplot as plt

def get_structuring_elements(mu=2):
    lambda_1 = mu * np.array([[0.5, 1.0, 0.5],
                        [1.0, 2.0, 1.0],
                        [0.5, 1.0, 0.5]])
    
    lambda_2 = mu * np.array([[0,   0.5, 0],
                        [0.5, 0.5, 0.5],
                        [0,   0.5, 0]])
    
    lambda_3 = mu * np.array([[0, 0,   0],
                        [0.5, 0.5, 0.5],
                        [0, 0,   0]])
    
    return lambda_1, lambda_2, lambda_3


def morphological_opening(image, selem):
    """
    O_lambda(x) = E_lambda(D_lambda(x))
    """
    dilated = grey_dilation(image, footprint=selem)
    eroded = grey_erosion(dilated, footprint=selem)
    return eroded

def MSANM(cA):
    """
    Parameters: low-frequency component cA
    Returns: edge image E_d
    """
    cA = img_as_float(cA)

    lambda_1, lambda_2, lambda_3 = get_structuring_elements(mu=2)

    # Dλ1
    d1 = grey_dilation(cA, footprint=lambda_1)

    # Eλ2(Dλ1)
    e2 = grey_erosion(d1, footprint=lambda_2)

    # Oλ3(Eλ2(Dλ1)) = Eλ3(Dλ3(Eλ2(Dλ1)))
    o3 = morphological_opening(e2, lambda_3)

    # Eλ3(Eλ2(Dλ1))
    e3 = grey_erosion(e2, footprint=lambda_3)


    E_d = o3 - e3
    return E_d

def visualize_MSANM(E_d):
    plt.figure(figsize=(6, 6))
    plt.imshow(E_d, cmap='gray')
    plt.title('Edge image $E_d $obtained by \n the multi-structure anti-noise morphological operator')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
