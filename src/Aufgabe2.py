import layer
import numpy as np
from math import sqrt

from src.GraphUtils import showColorMap

model = layer.Model([
    layer.InputLayer((2,)),
    layer.DenseLayer((4,),
                     # fixed_values=True,
                     # weights=np.array([
                     #     [25.0, 0.0, 25.0],
                     #     [-25.0, 0.0, 25.0],
                     #     [0.0, 25.0, 25.0],
                     #     [0.0, -25.0, 25.0],
                     # ])
                     ),
    layer.DenseLayer((1,),
                     # fixed_values=True,
                     # weights=np.array([[25, 25, 25, 25, -200]], dtype='float64')
                     )
])


def delta_learning():
    while (True):
        # model.visualize(2)
        showColorMap(model, -2, 2, 0.05)

        # train
        for x in np.arange(-2, 2, 0.05):
            for y in np.arange(-2, 2, 0.05):
                model.delta_learning(2, np.array([x, y]), np.array([0.8 if sqrt(x * x + y * y) < 1 else 0]), 0.05)


def backpropagation(i):
    while (True):
        # model.visualize(2)
        showColorMap(model, -2, 2, 0.05)

        # train
        for x in np.arange(-2, 2, 0.05):
            for y in np.arange(-2, 2, 0.05):
                model.backpropagation(np.array([x, y]), np.array([0.8 if sqrt(x * x + y * y) < 1 else 0]), 0.05)


backpropagation(0)
