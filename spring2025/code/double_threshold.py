import numpy as np
from skimage.morphology import binary_dilation

def double_threshold(nms, high_ratio=0.75, low_ratio=0.55):
    Hth = nms.max() * high_ratio
    Lth = Hth * low_ratio

    strong = (nms >= Hth).astype(np.uint8)
    weak = ((nms >= Lth) & (nms < Hth)).astype(np.uint8)

    result = np.copy(strong)
    prev = np.zeros_like(nms, dtype=np.uint8)

    while not np.array_equal(prev, result):
        prev = result.copy()
        dilated = binary_dilation(result)
        result = np.where((dilated & weak), 1, result)

    return result

import numpy as np
from scipy.ndimage import binary_dilation

def double_threshold_strict(nms, lambda_param=0.6, mu_param=0.5):
    flat = np.sort(nms.flatten())
    M = len(flat)


    H_idx = int(lambda_param * M)
    L_idx = int(mu_param * lambda_param * M)


    H_idx = min(max(H_idx, 0), M - 1)
    L_idx = min(max(L_idx, 0), M - 1)

    Hth = flat[H_idx]
    Lth = flat[L_idx]

    print(f"H_idx = {H_idx}, Hth = {Hth}")
    print(f"L_idx = {L_idx}, Lth = {Lth}")
    print(f"Strong count: {np.sum(nms >= Hth)}")
    print(f"Weak count: {(np.sum((nms >= Lth) & (nms < Hth)))}")


    strong = (nms >= Hth).astype(np.uint8)
    weak = ((nms >= Lth) & (nms < Hth)).astype(np.uint8)

    result = np.copy(strong)
    prev = np.zeros_like(nms, dtype=np.uint8)

    while not np.array_equal(prev, result):
        prev = result.copy()
        dilated = binary_dilation(result)
        result = np.where((dilated & weak), 1, result)

    return result
