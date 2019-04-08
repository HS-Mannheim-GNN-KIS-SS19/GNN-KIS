import layer
import numpy as np

from src.Utils import showColorMap

array = [
    # [-100, 0, 0],
    # [100, 0, 100],
    # [-100, 0, 100],
    # [100, 0, -100],
    # [-100, 0, -100],

    [100, 0, 100],
    [-100, 0, 100],
    [0, 100, 100],
    [0, -100, 100],
]

for i in range(len(array)):
    model = layer.Model([
        layer.InputLayer((2,)),
        layer.DenseLayer((1,),
                         fixed_values=True,
                         weights=np.array([
                             array[i]
                         ])),
    ])

    showColorMap(model, -2, 2, 0.1)
