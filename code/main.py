from LOSW_decomposition import losw_decompose
from LOSW_decomposition import visualize_decomposition
from MSANM import MSANM
from MSANM import visualize_MSANM
from SUAMM import SUAMM
from SUAMM import visualize_SUAMM

from skimage import io, color
import matplotlib.pyplot as plt

def visualize_result(image, filename_decomposition, filename_MSANM, filename_SUAMM):
    cA, cH, cV, cD = losw_decompose(image)
    visualize_decomposition(image, cA, cH, cV, cD, filename=filename_decomposition)
    E_d = MSANM(cA)
    visualize_MSANM(E_d, filename=filename_MSANM)
    (CH_prime, CV_prime, CD_prime) = SUAMM(E_d, cH, cV, cD)
    visualize_SUAMM(CH_prime, CV_prime, CD_prime, filename=filename_SUAMM)


image_1 = io.imread('examples/triangle1.png', as_gray=True)
visualize_result(image=image_1, filename_decomposition="results/LOSW_decomposition/LOSW_d_tr1.png", filename_MSANM="results/MSANM/MSANM_tr1.png", filename_SUAMM="results/SUAMM/SUAMM_tr1.png")

image_2 = io.imread('examples/triangle2.png', as_gray=True)
visualize_result(image=image_2, filename_decomposition="results/LOSW_decomposition/LOSW_d_tr2.png", filename_MSANM="results/MSANM/MSANM_tr2.png", filename_SUAMM="results/SUAMM/SUAMM_tr2.png")

image_3 = io.imread('examples/square1.png', as_gray=True)
visualize_result(image=image_3, filename_decomposition="results/LOSW_decomposition/LOSW_d_sq1.png", filename_MSANM="results/MSANM/MSANM_sq1.png", filename_SUAMM="results/SUAMM/SUAMM_sq1.png")

image_4 = io.imread('examples/square2.png', as_gray=True)
visualize_result(image=image_4, filename_decomposition="results/LOSW_decomposition/LOSW_d_sq2.png", filename_MSANM="results/MSANM/MSANM_sq2.png", filename_SUAMM="results/SUAMM/SUAMM_sq2.png")

image_5 = io.imread('examples/fox.png', as_gray=True)
visualize_result(image=image_5, filename_decomposition="results/LOSW_decomposition/LOSW_d_fox.png", filename_MSANM="results/MSANM/MSANM_fox.png", filename_SUAMM="results/SUAMM/SUAMM_fox.png")
