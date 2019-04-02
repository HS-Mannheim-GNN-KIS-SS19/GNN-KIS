import numpy as np
from io import StringIO


def readMatrix(path):
    return np.loadtxt(path, usecols=range(3))


print(readMatrix("Test.txt"))
