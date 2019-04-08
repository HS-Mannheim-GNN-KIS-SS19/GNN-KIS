import matplotlib.pyplot as plt
import numpy as np


def generateXYDataset(function, min, max, step):
    hstack = []
    for x in np.arange(min, max, step):
        vstack = []
        for y in np.arange(min, max, step):
            vstack.append(function(np.array([x, y])))
        hstack.append(np.vstack(vstack))
    return np.hstack(hstack)


def showColorMap(model, min, max, step):
    # calc array
    out = generateXYDataset(model.run, min, max, step)

    # plt
    plt.pcolor(np.arange(min, max, step), np.arange(min, max, step), out)
    plt.colorbar()
    plt.show()
