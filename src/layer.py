import abc

import matplotlib.pyplot as plt
import numpy as np


class Layer:
    def __init__(self, shape):
        self.shape = shape
        self.input_shape = None

    def attach(self, prev_layer):
        self.input_shape = prev_layer.shape

    @abc.abstractmethod
    def run(self, input):
        pass

    def assert_shape(self, shape):
        if shape != self.shape:
            raise AssertionError("invalid input shape: {} is not the same as {}".format(shape, self.shape))

    def assert_input_shape(self, shape):
        if shape != self.input_shape:
            raise AssertionError("invalid input shape: {} is not the same as {}".format(shape, self.input_shape))

    def visualize(self):
        pass


class DenseLayer(Layer):
    def __init__(self, shape):
        if len(shape) != 1:
            raise AssertionError("width.len has to be 1 was {}".format(shape.ndim))
        super().__init__(shape)
        self.weights = None

    def attach(self, prev_layer):
        if len(prev_layer.shape) != 1:
            raise AssertionError("Inputlayer width.len has to be 1 was {}".format(prev_layer.shape.ndim))
        super(DenseLayer, self).attach(prev_layer)
        self.weights = np.ones((prev_layer.shape[0], self.shape[0]))

    def run(self, input):
        self.assert_input_shape(input.shape)
        return np.dot(input, self.weights)

    def visualize(self):
        plt.imshow(self.weights, interpolation='nearest')
        plt.colorbar()
        plt.show()


class InputLayer(Layer):
    def __init__(self, shape):
        super().__init__(shape)

    def attach(self, prev_layer):
        super(InputLayer, self).attach(prev_layer)

    def run(self, input):
        self.assert_shape(input.shape)
        return input


class Model:
    def __init__(self, layers):
        if len(layers) <= 0:
            raise AssertionError("Layer count has to be at least 1")

        self.layers = layers

        prev_layer = layers[0]
        for i in range(1, len(layers)):
            next_layer = layers[i]
            next_layer.attach(prev_layer)
            prev_layer = next_layer

    def run(self, input):
        curr_state = input
        for i in range(len(self.layers)):
            curr_state = self.layers[i].run(curr_state)
        return curr_state

    def visualize(self, id):
        self.layers[id].visualize()
