import matplotlib.pyplot as plt

def morphological_reconstruction(E_d, CH_prime, CV_prime, CD_prime):
    """
    Parameters:
    ------------
    E_d : ndarray
        
    CH_prime, CV_prime, CD_prime : ndarray


    Returns:
    --------
    R : ndarray
        Restored image
    """
    return E_d + CH_prime + CV_prime + CD_prime

def visualize_morphologically_reconstructed(R, filename):
    plt.imshow(R, cmap='gray')
    plt.title("Restored image")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()