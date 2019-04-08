from time import sleep

import layer
import matplotlib.pyplot as plt
import numpy as np
# model
from math import sqrt

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


def generateXYDataset(function, min, max, step):
    hstack = []
    for x in np.arange(min, max, step):
        vstack = []
        for y in np.arange(min, max, step):
            vstack.append(function(np.array([x, y])))
            # vstack.append(0.8 if sqrt(x * x + y * y) < 1 else 0)
        hstack.append(np.vstack(vstack))
    return np.hstack(hstack)


def showColorMap(min, max, step):
    # calc array
    out = generateXYDataset(model.run, min, max, step)
    # out = generateXYDataset(lambda xy: 0.8 if sqrt(xy[0] * xy[0] + xy[1] * xy[1]) < 1 else 0, -5, 5, 0.1)

    # plt
    plt.pcolor(np.arange(min, max, step), np.arange(min, max, step), out)
    # plt.imshow(out, interpolation='bilinear')
    plt.colorbar()
    plt.show()


while (True):
    # model.visualize(2)

    # train
    for x in np.arange(-2, 2, 0.05):
        for y in np.arange(-2, 2, 0.05):
            model.train(2, np.array([x, y]), np.array([0 if sqrt(x * x + y * y) < 1 else 0.8]), 0.01)

    showColorMap(-5, 5, 0.1)

    sleep(0.25)
