from LOSW_decomposition import losw_decompose
from LOSW_decomposition import plot_decomposition
from MSANM import MSANM
from MSANM import visualize_MSANM

from skimage import io, color
import matplotlib.pyplot as plt

image = io.imread('example_triangle.png', as_gray=True)

cA, cH, cV, cD = losw_decompose(image)

plot_decomposition(image, cA, cH, cV, cD)

E_d = MSANM(cA)

visualize_MSANM(E_d)