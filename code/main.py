import LOSW_decomposition

image = io.imread('example_triangle.png', as_gray=True)

cA, cH, cV, cD = losw_decompose(image)

plot_decomposition(image, cA, cH, cV, cD)