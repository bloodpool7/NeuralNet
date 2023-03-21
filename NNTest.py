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

l1 = Layer(3, 3)
l1.set_weight(l1weights)
l1.set_bias(l1bias)

l2 = Layer(3, 3)
l2.set_weight(l2weights)
l2.set_bias(l2bias)

l3 = Layer(3, 2)
l3.set_weight(l3weights)
l3.set_bias(l3bias)

a1 = ReLU()
a2 = ReLU()
a3 = Softmax()

loss1 = Mean_Squared_Error()

optimizer = Adam()

# Using Model Object

model = Model(
    layers = [l1, l2, l3],
    activations = [a1, a2, a3],
    loss = loss1,
    optimizer = optimizer
)

print(f"model obj: ")
print(model.predict(inputs, targets))
print(model.loss)

print()

model.backward(targets)

print(model.predict(inputs, targets))
print(model.loss)

#Without Model Object (control)

print(f"no model obj: ")
l1 = Layer(3, 3)
l1.set_weight(l1weights)
l1.set_bias(l1bias)

l2 = Layer(3, 3)
l2.set_weight(l2weights)
l2.set_bias(l2bias)

l3 = Layer(3, 2)
l3.set_weight(l3weights)
l3.set_bias(l3bias)

a1 = ReLU()
a2 = ReLU()
a3 = Softmax()

loss1 = Mean_Squared_Error()

optimizer = Adam()

l1.forward(inputs)
a1.forward(l1.outputs)
l2.forward(a1.outputs)
a2.forward(l2.outputs)
l3.forward(a2.outputs)
a3.forward(l3.outputs)

loss_out = loss1.calculate(a3.outputs, targets)

print(a3.outputs)
print(loss_out)
print()

loss1.backward(a3.outputs, targets)
a3.backward(loss1.dinputs)
l3.backward(a3.dinputs)
a2.backward(l3.dinputs)
l2.backward(a2.dinputs)
a1.backward(l2.dinputs)
l1.backward(a1.dinputs)

optimizer.pre_param_updates()
optimizer.update_params(l1)
optimizer.update_params(l2)
optimizer.update_params(l3)
optimizer.post_param_updates()

l1.forward(inputs)
a1.forward(l1.outputs)
l2.forward(a1.outputs)
a2.forward(l2.outputs)
l3.forward(a2.outputs)
a3.forward(l3.outputs)

loss_out = loss1.calculate(a3.outputs, targets)

print(a3.outputs)
print(loss_out)

# X, y = mnist_load_data()
# X_valid, y_valid = mnist_load_test()

# layer1 = Layer(784, 200)
# layer2 = Layer(200, 10)

# activation1 = ReLU()
# loss_activation2 = Softmax_Entropy()

# optimizer = Adam()

# X, y = shuffle_data(X, y, 128)
# print("Beginning Training... ")
# start = time.time()
# for epoch in range(21):
#     accuracies = []
#     for i in range(len(X)):
#         layer1.forward(X[i])
#         activation1.forward(layer1.outputs)

#         layer2.forward(activation1.outputs)
#         loss = loss_activation2.forward(layer2.outputs, y[i])


#         predictions = np.argmax(loss_activation2.outputs, axis=1)
#         if len(y[i].shape) == 2:
#             y[i] = np.argmax(y, axis=1) 
#         accuracies.append(np.mean(predictions==y[i]))

#         loss_activation2.backward(loss_activation2.outputs, y[i])
#         layer2.backward(loss_activation2.dinputs)
#         activation1.backward(layer2.dinputs)
#         layer1.backward(activation1.dinputs)

#         optimizer.pre_param_updates()
#         optimizer.update_params(layer1)
#         optimizer.update_params(layer2)
#         optimizer.post_param_updates()
#     if not epoch % 1:
#         print(f'epoch: {epoch}, ' +
#               f'acc: {np.mean(accuracies):.3f}, ' + 
#               f'loss: {loss:.3f}, ' + 
#               f'lr: {optimizer.learning_rate:.5f}')

# end = time.time()

# print(f"Finished Training in : {end - start:.3f} seconds")
# print("Starting Testing...")

# layer1.forward(X_valid)
# activation1.forward(layer1.outputs)

# layer2.forward(activation1.outputs)
# loss = loss_activation2.forward(layer2.outputs, y_valid)


# predictions = np.argmax(loss_activation2.outputs, axis=1)
# if len(y_valid.shape) == 2:
#     y_valid = np.argmax(y, axis=1) 
# accuracy = np.mean(predictions==y_valid)

# print(f'acc: {accuracy:.3f}, ' + 
#       f'loss: {loss:.3f}, ')