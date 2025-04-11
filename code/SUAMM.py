import numpy as np

def suamm(Ed, cH, cV, cD):
    H, W = Ed.shape

    theta = np.arctan2(cV, cH)

    delta = np.mod(theta, 2 * np.pi)

    T_delta = 0.5 * (np.max(delta) + np.min(delta))

    theta_H = ((theta >= 0) & (theta < np.pi/8)) | ((theta >= 15*np.pi/8) & (theta < 2*np.pi)) | \
              ((theta >= 7*np.pi/8) & (theta < 9*np.pi/8))

    theta_V = ((theta >= 3*np.pi/8) & (theta < 5*np.pi/8)) | ((theta >= 11*np.pi/8) & (theta < 13*np.pi/8))

    theta_D = ((theta >= np.pi/8) & (theta < 3*np.pi/8)) | ((theta >= 9*np.pi/8) & (theta < 11*np.pi/8)) | \
              ((theta >= 5*np.pi/8) & (theta < 7*np.pi/8)) | ((theta >= 13*np.pi/8) & (theta < 15*np.pi/8))
    
    mask = ((theta_H | theta_V | theta_D) & (delta > T_delta))

    CH_prime = np.where(mask, cH, 0)
    CV_prime = np.where(mask, cV, 0)
    CD_prime = np.where(mask, cD, 0)

    return CH_prime, CV_prime, CD_prime
