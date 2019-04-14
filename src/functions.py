import numpy as np
from scipy.special import expit


def sigmoid(x):
    # use sigmoid function from scipy
    return expit(x)
   #return 1 / (1 + np.exp(-x/8.))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


relu = np.vectorize(lambda x: x if x > 0 else 0)

relu_derivative = np.vectorize(lambda x: 1.0 if x > 0 else 0)
