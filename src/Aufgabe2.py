import layer
import matplotlib.pyplot as plt
import numpy as np

# model
model = layer.Model([
    layer.InputLayer((2,)),
    layer.DenseLayer((4,)),
    layer.DenseLayer((1,))
])

# calc array
out = np.hstack([np.array([model.run(np.array([x, y])) for x in range(-5, 5)]) for y in range(-5, 5)])

# plt
plt.imshow(out, interpolation='bilinear')
plt.colorbar()
plt.show()
