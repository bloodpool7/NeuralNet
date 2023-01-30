import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from Model import *

nnfs.init()

X, y = spiral_data(100, 3)

layer1 = Layer_Dense(2, 64)
layer2 = Layer_Dense(64, 3)

activation1 = Activation_ReLU()
loss_activation2 = Softmax_Entropy()

optimizer = Optimizer_SGD(learning_rate = 0.85)


for epoch in range(10001):
    layer1.forward(X)
    activation1.forward(layer1.outputs)

    layer2.forward(activation1.outputs)

    loss = loss_activation2.forward(layer2.outputs, y)

    predictions = np.argmax(loss_activation2.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}')
    
    loss_activation2.backward(loss_activation2.outputs, y)
    layer2.backward(loss_activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.update_params(layer1)
    optimizer.update_params(layer2)



