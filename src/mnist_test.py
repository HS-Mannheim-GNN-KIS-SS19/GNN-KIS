import layer
import numpy as np
from src.GraphUtils import showColorMap

model = layer.Model([
    layer.InputLayer((784,)),  # 28 x 28 images
    layer.DenseLayer((512,)),
    layer.DenseLayer((256,)),
    layer.DenseLayer((128,)),
    layer.DenseLayer((10,))
])


def backpropagation(i):
    count = 0
    while True:
        if count == i:
            # model.visualize(2)

            count = 0

        # train
        for x in np.arange(-2, 2, 0.05):
            for y in np.arange(-2, 2, 0.05):
                pass


        count += 1

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
    for input_val, target in zip(all_x, all_y):
        model.backpropagation(input_val, target, 0.05)

        if count % draw_each_i == 0:
            showColorMap(model, 0, 9, 0.05)
        count += 1

a = np.array([1,2,1])
b = np.array([2,2,3])
print(np.dot(a,b))
number_recognize(800)
