import numpy as np
import matplotlib.pyplot as plt
from Model import *
from ReadData import *
import time

#weights & biases initialized the same for all layers for testing
# l1weights = (0.5 * np.random.randn(3, 3)).tolist()
# l2weights = (0.5 * np.random.randn(3, 3)).tolist()
# l3weights = (0.5 * np.random.randn(3, 2)).tolist()

# l1bias = np.array([[0, 0, 0]]).tolist()
# l2bias = np.array([[0, 0, 0]]).tolist()
# l3bias = np.array([[0, 0]]).tolist()

# weights = [l1weights, l2weights, l3weights]
# biases = [l1bias, l2bias, l3bias]

# inputs = np.array([[1, 2, 3]])

# targets = np.array([1])

# validation_inputs = np.array([[1, 2, 3]])
# valid_targets = np.array([1])

# #model object will use these layers
# l1 = Layer(3, 3)
# l1.set_weight(l1weights)
# l1.set_bias(l1bias)

# l2 = Layer(3, 3)
# l2.set_weight(l2weights)
# l2.set_bias(l2bias)

# l3 = Layer(3, 2)
# l3.set_weight(l3weights)
# l3.set_bias(l3bias)

# a1 = ReLU()
# a2 = ReLU()
# a3 = Softmax()

# loss = CategoricalCrossEntroy()

# optimizer = Adam()

# #non model object will use these layers
# l1_copy = Layer(3, 3)
# l1_copy.set_weight(l1weights)
# l1_copy.set_bias(l1bias)

# l2_copy = Layer(3, 3)
# l2_copy.set_weight(l2weights)
# l2_copy.set_bias(l2bias)

# l3_copy = Layer(3, 2)
# l3_copy.set_weight(l3weights)
# l3_copy.set_bias(l3bias)

#Testing the train function in Model

l1 = Layer(784, 128)
l2 = Layer(128, 64)
l3 = Layer(64, 10)

a1 = ReLU()
a2 = ReLU()
a3 = Softmax()

loss = CategoricalCrossEntroy()

optimizer = Adam()

model = Model(
    layers = [l1, l2, l3],
    activations = [a1, a2, a3],
    loss = loss,
    optimizer = optimizer
)

X_train, y_train = mnist_load_data()

X_train, y_train = shuffle_data(X_train, X_train, batch_size = 128)

print(len(X_train[0].shape))

X_valid, y_valid = mnist_load_test()

