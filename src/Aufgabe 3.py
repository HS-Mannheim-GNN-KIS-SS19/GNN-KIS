import layer
import matplotlib.pyplot as plt
import numpy as np

PICTURE_SIZE = 28 * 28
NUMBER_NEURONS = 10

LEARN_RATE = 0.05
SAMPLING_AMOUNT = 1

VISUALIZE_PROGRESS_EACH_I = 100

model = layer.Model([
    layer.InputLayer((PICTURE_SIZE + NUMBER_NEURONS,)),
    layer.DenseLayer((PICTURE_SIZE + NUMBER_NEURONS,)),
])


# Will show output after sampling in the colormap to the right
# Predicted and expected numbers as well as accuracy is currently shown in the console
def visualize(original_image, target_number, output, right_amount, count):
    # picture without the 10 extra neurons
    pic = output[:PICTURE_SIZE]
    # only the 10 extra neurons
    predicted = output[PICTURE_SIZE:PICTURE_SIZE + NUMBER_NEURONS]

    plt.pcolormesh(np.arange(0, 28, 1), np.arange(0, 28, 1), np.flipud(np.array(pic).reshape((28, 28))))
    plt.colorbar()
    plt.show()

    # Print each output neuron
    for i, o in enumerate(predicted):
        print("{}    {:1.6f}".format(i, o))

    print("Predicted:  {}   Expected:  {} for this picture".format(predicted.tolist().index(max(predicted)),
                                                                   target_number))
    print("recognized {}% correctly in the last {} pictures".format((right_amount / count) * 100,
                                                                    VISUALIZE_PROGRESS_EACH_I))
    print("--------------")


def run_restricted_boltzmann_machine():
    from mlxtend.data import loadlocal_mnist

    images, corresponding_numbers = loadlocal_mnist(
        images_path='../trainingdata/train-images.idx3-ubyte',
        labels_path='../trainingdata/train-labels.idx1-ubyte')

    count = 0
    right = 0
    for image, image_corresponding_number in zip(images, corresponding_numbers):
        target_vec = np.zeros((10,))
        target_vec[image_corresponding_number] = 255

        # Doesn't do the trick here... what a surprise. Still wanted to see what happens in Backpropagation xD
        # output = model.backpropagation(np.append(image, np.zeros((10,))), np.append(image, target_vec), 0.01)
        output = model.gibbs_sampling(1, np.append(image, target_vec), LEARN_RATE, SAMPLING_AMOUNT)

        count += 1

        # Visualization
        if count % VISUALIZE_PROGRESS_EACH_I == 0:
            visualize(image, image_corresponding_number, output, right, count)

        output_numbers = output[PICTURE_SIZE:PICTURE_SIZE + NUMBER_NEURONS]

        if output_numbers.tolist().index(max(output_numbers)) == image_corresponding_number:
            right += 1


run_restricted_boltzmann_machine()
