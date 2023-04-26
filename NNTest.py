import numpy as np
import matplotlib.pyplot as plt
from Model import *
from ReadData import *
import time

#weights & biases initialized the same for all layers for testing
l1weights = (0.5 * np.random.randn(3, 3)).tolist()
l2weights = (0.5 * np.random.randn(3, 3)).tolist()
l3weights = (0.5 * np.random.randn(3, 2)).tolist()

l1bias = np.array([[0, 0, 0]]).tolist()
l2bias = np.array([[0, 0, 0]]).tolist()
l3bias = np.array([[0, 0]]).tolist()

weights = [l1weights, l2weights, l3weights]
biases = [l1bias, l2bias, l3bias]

inputs = np.array([[1, 2, 3]])

targets = np.array([1])

validation_inputs = np.array([[1, 2, 3]])
valid_targets = np.array([1])

l1 = Layer(3, 3)
l1.set_weight(l1weights)
l1.set_bias(l1bias)

l2 = Layer(3, 3)
l2.set_weight(l2weights)
l2.set_bias(l2bias)

l3 = Layer(3, 3)
l3.set_weight(l3weights)
l3.set_bias(l3bias)


#Testing the train function in Model