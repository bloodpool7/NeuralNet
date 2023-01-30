import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from Model import *
from ReadData import *

nnfs.init()

X, y = spiral_data(100, 3)

layer1 = Layer_Dense(2, 64)
layer2 = Layer_Dense(64, 3)

activation1 = Activation_ReLU()
loss_activation2 = Softmax_Entropy()

optimizer = Optimizer_SGD(learning_rate = 1.5, decay = 0.0002, momentum = 0.9)

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
              f'loss: {loss:.3f}, ' + 
              f'lr: {optimizer.current_learning_rate:.5f}')
    
    loss_activation2.backward(loss_activation2.outputs, y)
    layer2.backward(loss_activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()



