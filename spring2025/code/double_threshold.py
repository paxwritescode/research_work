import numpy as np
from skimage.morphology import binary_dilation

def double_threshold(nms, high_ratio=0.55, low_ratio=0.45):
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
