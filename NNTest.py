import numpy as np
import matplotlib.pyplot as plt
from Model_GPU import *
from ReadData import *
import time
import pandas as pd

data = pd.read_csv("Data/mnist_train.csv")
inputs = data.iloc[:, 1:].to_numpy() / 255
labels = data.iloc[:, 0].to_numpy() 

l1 = Layer(784, 256)
a1 = ReLU()
l2 = Layer(256, 10)
a2 = Softmax_Entropy()

inputs = np.array(np.array_split(inputs, 600))
labels = np.array(np.array_split(labels, 600))

sections = [l1, a1, l2]

for i in range(inputs):
    input_batch = cp.from_numpy(inputs[i])
    label_batch = cp.from_numpy(labels[i])

    previous = input_batch
    for section in sections:
        section.forward(previous)
        previous = section.outputs

    loss = a2.forward(previous, label_batch)
    a2.backward(a2.outputs, label_batch)

    previous = a2.dinputs
    for section in reversed(sections):
        section.backward(previous)
        previous = section.dinputs
    
    
