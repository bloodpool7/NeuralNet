import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from Model import *
from ReadData import *
import time

nnfs.init()

X, y = iris_load_data()


layer1 = Layer_Dense(4, 16)
layer2 = Layer_Dense(16, 16)
layer3 = Layer_Dense(16, 3)

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
loss_activation3 = Softmax_Entropy()

optimizer = Optimizer_SGD(learning_rate = 0.01)

start = time.time()
for epoch in range(101):
    layer1.forward(X)
    activation1.forward(layer1.outputs)

    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)

    layer3.forward(activation2.outputs)
    loss = loss_activation3.forward(layer3.outputs, y)

    predictions = np.argmax(loss_activation3.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y)

    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}, ' + 
              f'lr: {optimizer.current_learning_rate:.5f}')
    
    loss_activation3.backward(loss_activation3.outputs, y)
    layer3.backward(loss_activation3.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.update_params(layer3)
    optimizer.post_update_params()

end = time.time()

print(f"Finished in : {end - start:.3f} seconds")