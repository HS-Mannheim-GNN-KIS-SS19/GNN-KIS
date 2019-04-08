from time import sleep

import layer
import numpy as np
from math import sqrt

from src.GraphUtils import showColorMap

model = layer.Model([
    layer.InputLayer((2,)),
    layer.DenseLayer((4,),
                     fixed_values=True,
                     weights=np.array([[1, 0, 2.8],
                                       [-1, 0, 2.8],
                                       [0, 1, 2.8],
                                       [0, -1, 2.8]])
                     ),
    layer.DenseLayer((1,),
                     # fixed_values=True,
                     weights=np.array([[1, 1, 1, 1, 0]], dtype='float64')
                     )
])

while (True):
    # model.visualize(2)

    # train
    for x in np.arange(-2, 2, 0.05):
        for y in np.arange(-2, 2, 0.05):
            model.train(2, np.array([x, y]), np.array([0 if sqrt(x * x + y * y) < 1 else 0.8]), 0.01)

    showColorMap(-5, 5, 0.1)

    sleep(0.25)
