import layer
import numpy as np


def solution_to_5():
    layer.RecurrentLayer(size=2,
                         start_values=np.array([0.770719, 0.00978897]),
                         weights=np.array([
                             [4.0, 6.0, -4.0],
                             [-10.0, 0.0, 4.0],
                         ])
                         ).run_times(4, print_flag=True)


# Altering weights so output converges to a fixed point
def solution_to_6():
    factor = 1 / 10
    layer.RecurrentLayer(size=2,
                         start_values=np.array([0.770719, 0.00978897]),
                         weights=np.array([
                             [4.0, 6.0, -4.0],
                             [-10.0, 0.0, 4.0],
                         ]) * factor
                         ).run_times(20, print_flag=True)


solution_to_5()
print("Altering weights so output converges to a fixed point")
solution_to_6()
