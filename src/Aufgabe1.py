import matplotlib.pyplot as plt
import numpy as np

count = 10
step = 0.01
start = [-7, -0.2, 8]

for i in range(len(start)):
    result = []
    x = start[i]
    for i in range(int(count / step)):
        result.append(x)
        x += (x - (x * x * x)) * step

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, count, step), np.array(result))

    ax.set(xlabel='x', ylabel='y', title="start: {}".format(start))
    ax.grid()

    plt.show()
