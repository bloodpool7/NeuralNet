import numpy as np
import matplotlib.pyplot as plt
from Model import *
from ReadData import *
import time


X, y = mnist_load_data()


layer1 = Layer_Dense(784, 300)
layer2 = Layer_Dense(300, 300)
layer3 = Layer_Dense(300, 10)

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
loss_activation3 = Softmax_Entropy()

optimizer = Optimizer_SGD(learning_rate = 0.05, momentum = 0.5, decay = 0.0001)

start = time.time()
X, y = shuffle_data(X, y, 128)
for epoch in range(10):
    for i in range(len(X)):
        layer1.forward(X[i])
        activation1.forward(layer1.outputs)

        layer2.forward(activation1.outputs)
        activation2.forward(layer2.outputs)

        layer3.forward(activation2.outputs)
        loss = loss_activation3.forward(layer3.outputs, y[i])

        predictions = np.argmax(loss_activation3.outputs, axis=1)
        if len(y[i].shape) == 2:
            y[i] = np.argmax(y, axis=1) 
        accuracy = np.mean(predictions==y[i])

        
        loss_activation3.backward(loss_activation3.outputs, y[i])
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

    if not epoch % 1:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}, ' + 
              f'lr: {optimizer.current_learning_rate:.5f}')

end = time.time()

print(f"Finished in : {end - start:.3f} seconds")