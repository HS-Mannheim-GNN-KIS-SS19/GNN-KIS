import numpy as np


def readMatrix(path):
    return np.loadtxt(path, usecols=range(3))


print(readMatrix("Test.txt"))
