import abc

import matplotlib.pyplot as plt
import numpy as np


class Layer:
    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.input_shape = None

    def attach(self, prev_layer):
        self.input_shape = prev_layer.output_shape

    @abc.abstractmethod
    def run(self, input):
        pass

    def assert_output_shape(self, shape):
        if shape != self.output_shape:
            raise AssertionError("invalid input shape: {} is not the same as {}".format(shape, self.output_shape))

    def assert_input_shape(self, shape):
        if shape != self.input_shape:
            raise AssertionError("invalid input shape: {} is not the same as {}".format(shape, self.input_shape))

    def visualize(self):
        pass

    def train(self, input, target, learn_rate):
        raise AssertionError("This layer cannot be trained!")


class DenseLayer(Layer):
    def __init__(self, shape, fixed_values=False, weights=None):
        if len(shape) != 1:
            raise AssertionError("width.len has to be 1 was {}".format(shape.ndim))
        super().__init__(shape)
        self.fixed_values = fixed_values
        self.weights = weights

    def attach(self, prev_layer):
        if len(prev_layer.output_shape) != 1:
            raise AssertionError("Inputlayer width.len has to be 1 was {}".format(prev_layer.output_shape.ndim))
        super(DenseLayer, self).attach(prev_layer)

        if self.weights is None:
            self.weights = np.random.randn(self.output_shape[0], self.input_shape[0] + 1)
        else:
            if self.weights.shape != (self.output_shape[0], self.input_shape[0] + 1):
                raise AssertionError("manually set weights are not correctly sized: {} instead of required {}"
                                     .format(self.weights.shape, (self.output_shape[0], self.input_shape[0] + 1)))

    def run(self, input):
        self.assert_input_shape(input.shape)
        return sigmoid(np.dot(self.weights, np.append(input, 1)))

    def train(self, input, target, learn_rate):
        if self.fixed_values:
            return

        self.assert_input_shape(input.shape)
        self.assert_output_shape(target.shape)

        sum = np.dot(self.weights, np.append(input, 1))
        self.weights += -learn_rate * np.append(input, 1) * (sigmoid(sum) - target)  # * sigmoid_derivative(sum)

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
        self.assert_output_shape(input.shape)
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

    def visualize(self, layer_id):
        self.layers[layer_id].visualize()

    def train(self, layer_id, input, target, learn_rate):
        curr_state = input
        for i in range(layer_id):
            curr_state = self.layers[i].run(curr_state)
        self.layers[layer_id].train(curr_state, target, learn_rate)


def sigmoid(x):
    return 1 / (1 + np.exp(-x / 8.))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
