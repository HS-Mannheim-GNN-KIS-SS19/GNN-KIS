import numpy as np
from scipy.special import expit
import random

def sigmoid(x):
    return expit(x)


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def stochastic(x):
    if random.random() < x:
        return 1
    else:
        return 0


relu = np.vectorize(lambda x: x if x > 0 else 0)

relu_derivative = np.vectorize(lambda x: 1.0 if x > 0 else 0)
