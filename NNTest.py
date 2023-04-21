import numpy as np
import matplotlib.pyplot as plt
from Model import *
from ReadData import *
import time

#weights & biases initialized the same for all layers for testing


# Using Model Object

model = Model(
    # layers = [Layer(3, 3), Layer(3, 3), Layer(3, 2)],
    activations = [ReLU(), ReLU(), Softmax()],
    loss = Mean_Squared_Error(),
    optimizer = SGD(learning_rate = 0.1)
)

model.load_model()

inputs = np.array([[1, 2, 3]])

targets = np.array([0])

print(model.predict(inputs, targets))
print(model.loss)

#Without Model Object (control)


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