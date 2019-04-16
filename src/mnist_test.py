import layer
import matplotlib.pyplot as plt
import numpy as np

model = layer.Model([
    layer.InputLayer((784,)),  # 28 x 28 images
    layer.DenseLayer((512,)),
    layer.DenseLayer((256,)),
    layer.DenseLayer((128,)),
    layer.DenseLayer((10,))  # Outputs numbers from 0 - 9
])


def visualize(input_val, target_number, output):
    # Plot the original 28 x 28 image
    plt.pcolormesh(np.arange(0, 28, 1), np.arange(0, 28, 1), np.flipud(np.array(input_val).reshape((28, 28))))
    plt.colorbar()
    plt.show()

    # Print each output neuron
    for i, o in enumerate(output):
        print("{}    {:1.6f}".format(i, o))

    print("Predicted:  {}   Result:  {}".format(output.tolist().index(max(output)), target_number))
    print("--------------")


def number_recognize(draw_each_i):
    from mlxtend.data import loadlocal_mnist

    all_x, all_y = loadlocal_mnist(
        images_path='../trainingdata/train-images.idx3-ubyte',
        labels_path='../trainingdata/train-labels.idx1-ubyte')

    print('Dimensions: %s x %s' % (all_x.shape[0], all_x.shape[1]))
    print('Digits:  0 1 2 3 4 5 6 7 8 9')
    print('labels: %s' % np.unique(all_y))
    print('Class distribution: %s' % np.bincount(all_y))

    count = 0
    right, wrong = 0, 0
    for input_val, target in zip(all_x, all_y):
        # very important!
        # target must not be the correct number but instead
        # a vector with the height of the output layer where only the correct index is 1 (rest 0)
        target_vec = np.zeros((10,))
        target_vec[target] = 1

        if count % draw_each_i == 0:
            output = model.backpropagation(input_val, target_vec, 0.05)
            visualize(input_val, target, output)
        else:
            output = model.backpropagation(input_val, target_vec, 0.05)

        # lines below are just for testing
        if count > 20000:
            if output.tolist().index(max(output)) == target:
                right += 1
            else:
                wrong += 1

            if (right + wrong) % 150 == 0:
                print("recognized {}% correctly".format(right / (count - 20000)))
        count += 1


number_recognize(10000)
