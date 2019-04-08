from time import sleep

import layer
import numpy as np
from math import sqrt

from src.GraphUtils import showColorMap

model = layer.Model([
    layer.InputLayer((2,)),
    layer.DenseLayer((4,),
                     fixed_values=True,
                     weights=np.array([
                         [100, 0, 100],
                         [-100, 0, 100],
                         [0, 100, 100],
                         [0, -100, 100],
                     ])),
    layer.DenseLayer((1,),
                     # fixed_values=True,
                     # weights=np.array([[50, 50, 50, 50, -200]], dtype='float64')
                     )
])

while (True):
    # model.visualize(2)

    showColorMap(model, -2, 2, 0.1)

    # train
    for x in np.arange(-2, 2, 0.05):
        for y in np.arange(-2, 2, 0.05):
            model.train(2, np.array([x, y]), np.array([0.8 if sqrt(x * x + y * y) < 1 else 0]), 0.01)

    sleep(0.25)
