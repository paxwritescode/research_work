import numpy as np
import cv2

def non_maximum_suppression(esm):
    H, W = esm.shape

    grad_x = cv2.Sobel(esm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(esm, cv2.CV_64F, 0, 1, ksize=3)

    angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
    angle[angle < 0] += 180

    nms = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q, r = 255, 255 

            direction = angle[i, j]

            try:
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    q = esm[i, j + 1]
                    r = esm[i, j - 1]
                elif 22.5 <= direction < 67.5:
                    q = esm[i + 1, j - 1]
                    r = esm[i - 1, j + 1]
                elif 67.5 <= direction < 112.5:
                    q = esm[i + 1, j]
                    r = esm[i - 1, j]
                elif 112.5 <= direction < 157.5:
                    q = esm[i - 1, j - 1]
                    r = esm[i + 1, j + 1]
                if esm[i, j] >= q and esm[i, j] >= r:
                    nms[i, j] = esm[i, j]
                else:
                    nms[i, j] = 0.0

            except IndexError:
                pass

    return nms / np.max(nms) 