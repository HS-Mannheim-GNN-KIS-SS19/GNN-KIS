import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

import layer
from functions import *

PICTURE_SIZE = 28 * 28
NUMBER_NEURONS = 10

LEARN_RATE = 0.05
SAMPLING_AMOUNT = 1

VISUALIZE_PROGRESS_EACH_I = 500
learn_for_steps = 5000

model = layer.Model([
    layer.InputLayer((PICTURE_SIZE + NUMBER_NEURONS,)),
    layer.DenseLayer((PICTURE_SIZE - 200 + NUMBER_NEURONS,), function=sigmoid),
])


# Will show output after sampling in the colormap to the right
# Predicted and expected numbers as well as accuracy is currently shown in the console
def visualize(original_image, target_number, output, right_amount, count):
    # picture without the 10 extra neurons
    pic = output[:PICTURE_SIZE]
    # only the 10 extra neurons
    predicted = output[PICTURE_SIZE:PICTURE_SIZE + NUMBER_NEURONS]

    plt.pcolormesh(np.arange(0, 28, 1), np.arange(0, 28, 1),
                   np.flipud(np.array(original_image).reshape((28, 28))))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(np.arange(0, 28, 1), np.arange(0, 28, 1),
                   np.flipud(np.array(pic).reshape((28, 28))))
    plt.colorbar()
    plt.show()

    # Print each output neuron
    for i, o in enumerate(predicted):
        print("{}    {:1.6f}".format(i, o))

    print("Predicted:  {}   Expected:  {} for this picture".format(predicted.tolist().index(max(predicted)),
                                                                   target_number))
    print("recognized {:1.2f}% correctly in the last {} pictures".format((right_amount / count) * 100,
                                                                         VISUALIZE_PROGRESS_EACH_I))
    print("--------------")


def run_restricted_boltzmann_machine(visualize_while_learning, save_weights_each=0):
    images, corresponding_numbers = loadlocal_mnist(
        images_path='../trainingdata/train-images.idx3-ubyte',
        labels_path='../trainingdata/train-labels.idx1-ubyte')

    count = 0
    right = 0
    learn = True
    for image, image_corresponding_number in zip(images, corresponding_numbers):
        target_vec = np.zeros((10,))
        if learn:
            target_vec[image_corresponding_number] = 1
        for i in range(len(image)):
            if image[i] > 0:
                image[i] = 1
            else:
                image[i] = 0

        if learn:
            output = model.gibbs_sampling(1, np.append(
                image, target_vec), LEARN_RATE, SAMPLING_AMOUNT)
        else:
            output = model.reconstruct(1, np.append(image, target_vec))

        count += 1

        if save_weights_each > 0 and count % save_weights_each == 0:
            model.save_to_file('bolzman')

        if count == learn_for_steps:
            learn = False

        # Visualization
        if count % VISUALIZE_PROGRESS_EACH_I == 0 and visualize_while_learning:
            visualize(image, image_corresponding_number, output, right, count)
            count, right = 0, 0

        output_numbers = output[PICTURE_SIZE:PICTURE_SIZE + NUMBER_NEURONS]

        if output_numbers.tolist().index(max(output_numbers)) == image_corresponding_number:
            right += 1


run_restricted_boltzmann_machine(True, save_weights_each=0)
