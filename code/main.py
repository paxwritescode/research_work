from LOSW_decomposition import losw_decompose
from LOSW_decomposition import plot_decomposition

from skimage import io, color

image = io.imread('example_triangle.png', as_gray=True)

cA, cH, cV, cD = losw_decompose(image)

plot_decomposition(image, cA, cH, cV, cD)