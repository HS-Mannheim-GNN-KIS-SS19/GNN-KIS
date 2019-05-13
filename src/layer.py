import abc

import matplotlib.pyplot as plt
import numpy as np

from src.functions import sigmoid, sigmoid_derivative


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

    def delta_learning(self, input, target, learn_rate):
        raise AssertionError("This layer cannot be trained!")

    def backpropagation(self, layers, o, ek, target, learn_rate):
        raise AssertionError("This layer cannot be trained!")


class DenseLayer(Layer):
    def __init__(self, shape, function=sigmoid, function_derivative=sigmoid_derivative, fixed_values=False,
                 weights=None):
        if len(shape) != 1:
            raise AssertionError("width.len has to be 1 was {}".format(shape.ndim))
        super().__init__(shape)
        self.fixed_values = fixed_values
        self.weights = weights
        self.function = function
        self.function_derivative = function_derivative

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
        return self.function(np.dot(self.weights, np.append(input, 1)))

    def delta_learning(self, input, target, learn_rate):
        if self.fixed_values:
            return

        self.assert_input_shape(input.shape)
        self.assert_output_shape(target.shape)

        sum = np.dot(self.weights, np.append(input, 1))
        self.weights += -learn_rate * np.append(input, 1) * (
                self.function(sum) - target)  # * self.function_derivative(sum)
        # print(self.weights)

    def backpropagation(self, layers, o, ek, target, learn_rate):
        if self.function != sigmoid:
            raise AssertionError("backpropagation only works with sigmoid function! Not {}".format(self.function))

        # output vector of current layer
        oj = np.take(o, 0)
        # output matrix of all neurons
        o = np.delete(o, 0)
        # oi are always the inputs with added bias neuron
        oi = np.append(o[0], 1)

        # "each layer owns the weights of its inputs"
        if ek is None:
            # Backpropagation output layer
            ej = (oj - target) * oj * (1 - oj)
            self.weights += -learn_rate * np.mat(ej).T * oi

        else:
            # Backpropagation hidden layers
            # weights[:, :-1] deletes the last value/vector of the array/matrix (weight of bias neuron not needed here)
            ej = np.mat(ek) * (layers[len(o) + 1].weights[:, :-1] * oj * (1 - oj))
            self.weights += -learn_rate * ej.T * oi

        # Until it reached the input layer
        if len(o) > 1:
            layers[len(o) - 1].backpropagation(layers, o, ej, target, learn_rate)

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

    def save_to_file(self, filename_prefix):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer) or isinstance(layer, RecurrentLayer):
                np.savetxt('../saves/' + filename_prefix + '_layer' + str(i) + '.gz', layer.weights)

    def restore_from_file(self, filename_prefix):
        # starts at 1 because layer 0 is the input layer which has no weights
        for i in range(1, len(self.layers)):
            weights = np.loadtxt('../saves/' + filename_prefix + '_layer' + str(i) + '.gz')
            if self.layers[i].weights.shape == weights.shape:
                self.layers[i].weights = weights
            else:
                raise AssertionError(
                    "Error loading weights: Incompatible Shapes {} and {}".format(self.layers[i].weights, weights))

    def delta_learning(self, layer_id, input, target, learn_rate):
        curr_state = input
        for i in range(layer_id):
            curr_state = self.layers[i].run(curr_state)
        self.layers[layer_id].delta_learning(curr_state, target, learn_rate)

    def backpropagation(self, input, target, learn_rate):
        cur = input
        output = []
        for layer in self.layers:
            output.insert(0, layer.run(cur))
            cur = output[0]
        self.layers[- 1].backpropagation(self.layers, np.array(output), None, target, learn_rate)
        return np.array(output)[0]


class RecurrentLayer(Layer):

    def __init__(self, size, start_values, weights=None):
        super().__init__(size)
        self._model = Model([
            InputLayer((size,)),
            DenseLayer((size,),
                       weights=weights
                       ),
        ])
        self.state = start_values
        self.weights = self._model.layers[1].weights

    def run_times(self, i, print_flag=False, zfill=2):
        if print_flag:
            print("{}: {}".format(str(0).zfill(zfill), self.state))
        for t in range(i):
            self.run()
            if print_flag:
                print("{}: {}".format(str(t + 1).zfill(zfill), self.state))
        return self.state

    def run(self, new_state=None):
        if new_state is None:
            self.state = self._model.run(self.state)
        else:
            self.state = self._model.run(new_state)
        return self.state

    def visualize(self):
        self._model.visualize(1)

    def save_to_file(self, filename_prefix):
        self._model.save_to_file(filename_prefix)

    def restore_from_file(self, filename_prefix):
        self._model.restore_from_file(filename_prefix)
