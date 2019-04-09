import numpy as np


def read_matrix_from_file(path):
    return np.loadtxt(path, usecols=range(3))


def random_matrix(width, height):
    return np.random.randn(height, width)


print(read_matrix_from_file("Test.txt"))
