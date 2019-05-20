import layer
import matplotlib.pyplot as plt
import numpy as np

STEPS = 500
outputs = []


def visualize(output, length):
    fig, ax = plt.subplots()
    a1 = [x[0] for x in outputs]
    a2 = [x[1] for x in outputs]
    ax.plot(a1, a2)
    ax.grid()

    plt.show()


def solution_to_5():
    layer.RecurrentLayer(size=2,
                         function=np.tanh,
                         start_values=np.array([0.0, 0.0]),
                         weights=np.array([
                             [-4.0, 1.5, -3.37],
                             [-1.5, 0.0, 0.125],
                         ])
                         ).run_times(STEPS, output=lambda x: outputs.append(x))
    print(outputs)
    visualize(np.array(outputs), STEPS)


# Altering weights so output converges to a fixed point
# def solution_to_6():
#     factor = 1 / 10
#     layer.RecurrentLayer(size=2,
#                          start_values=np.array([0.770719, 0.00978897]),
#                          weights=np.array([
#                              [4.0, 6.0, -4.0],
#                              [-10.0, 0.0, 4.0],
#                          ]) * factor
#                          ).run_times(20, print_flag=True)
#

solution_to_5()
# print("Altering weights so output converges to a fixed point")
# solution_to_6()
