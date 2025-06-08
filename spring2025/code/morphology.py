import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes

def morphological_refinement(binary_edges, closing_size=3):
    structure = np.ones((closing_size, closing_size), dtype=np.uint8)

    closed = binary_closing(binary_edges, structure=structure)
    filled = binary_fill_holes(closed)

    return filled.astype(np.uint8)
