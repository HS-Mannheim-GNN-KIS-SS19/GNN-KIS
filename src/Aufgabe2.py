import layer
import numpy as np
from math import sqrt

from src.GraphUtils import showColorMap


model = layer.Model([
    layer.InputLayer((2,)),
    layer.DenseLayer((4,)),
    layer.DenseLayer((1,))
])


def delta_learning():
    while True:
        # model.visualize(2)
        showColorMap(model, -2, 2, 0.05)

        # train
        for x in np.arange(-2, 2, 0.05):
            for y in np.arange(-2, 2, 0.05):
                model.delta_learning(2, np.array([x, y]), np.array([0.8 if sqrt(x * x + y * y) < 1 else 0]), 0.05)


def backpropagation(i):
    count = 0
    while True:
        if count == i:
            # model.visualize(2)
            showColorMap(model, -2, 2, 0.05)
            count = 0

        # train
        for x in np.arange(-2, 2, 0.05):
            for y in np.arange(-2, 2, 0.05):
                model.backpropagation(np.array([x, y]), np.array([0.8 if sqrt(x * x + y * y) < 1 else 0]), 0.05)

        count += 1


backpropagation(8)
