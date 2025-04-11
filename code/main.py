from LOSW_decomposition import losw_decompose
from LOSW_decomposition import visualize_decomposition
from MSANM import MSANM
from MSANM import visualize_MSANM

from skimage import io, color
import matplotlib.pyplot as plt

def visualize_result(image, filename_decomposition, filename_MSANM):
    cA, cH, cV, cD = losw_decompose(image)
    visualize_decomposition(image, cA, cH, cV, cD, filename=filename_decomposition)
    E_d = MSANM(cA)
    visualize_MSANM(E_d, filename=filename_MSANM)


image_1 = io.imread('examples/example_triangle1.png', as_gray=True)
visualize_result(image=image_1, filename_decomposition="results/LOSW_decomposition_tr1.png", filename_MSANM="results/MSANM_res_tr1.png")

image_2 = io.imread('examples/example_triangle2.png', as_gray=True)
visualize_result(image=image_2, filename_decomposition="results/LOSW_decomposition_tr2.png", filename_MSANM="results/MSANM_res_tr2.png")

image_3 = io.imread('examples/example_square1.png', as_gray=True)
visualize_result(image=image_3, filename_decomposition="results/LOSW_decomposition_sq1.png", filename_MSANM="results/MSANM_res_sq1.png")
