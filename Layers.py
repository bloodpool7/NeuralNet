import numpy as np
import nnfs 
from nnfs.datasets import spiral_data, vertical_data
import time

nnfs.init()

class Layer_Dense:
    #parameters are the number of neurons by the number of features
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.5 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros(shape = (1, n_neurons))

    def set_weight(self, weight_in):
        self.weights = weight_in

    def set_bias(self, bias_in):
        self.bias = bias_in

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs =  np.dot(inputs, self.weights) + self.bias
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbias = np.sum(dvalues, axis = 0, keepdims = True)

class Activation_ReLU:
    #the inputs are meant to be the output of a forward pass of a dense layer
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    #the inputs are meant to be the output of a hidden layer typically using a ReLU activation function
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_inputs / np.sum(exp_inputs, axis = 1, keepdims = True)
        self.outputs = probabilities

class Loss:
    def calculate(self, outputs, y):
        sample_loss = self.forward(outputs, y)
        data_loss = np.mean(sample_loss)
        return data_loss

class CategoricalCrossEntroy(Loss):
    def forward(self, y_pred, y_target):
        self.inputs = np.array(y_pred)

        y_target = np.array(y_target)
        if (len(y_target.shape) == 1):
            self.targets = np.zeros_like(y_pred)
            self.targets[range(len(y_pred)), y_target] = 1
        else:
            self.targets = y_target

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        correct_confidences = np.sum(y_pred_clipped * self.targets, axis = 1)
        total_loss = -np.log(correct_confidences)
        return total_loss
    
    def backward(self, outputs, labels):
        pass

class Mean_Squared_Error(Loss):
    def forward(self, y_pred, y_target):
        self.inputs = y_pred

        y_target = np.array(y_target)
        if (len(y_target.shape) == 1):
            self.targets = np.zeros_like(y_pred)
            self.targets[range(len(y_pred)), y_target] = 1
        else:
            self.targets = y_target

        total_loss = (y_pred - self.targets) ** 2
            
        return total_loss

    def backward(self, outputs, labels):
        self.dinputs = np.copy((2 * outputs - 2 * labels)/len(outputs))


outputs = np.array([[0.3, 0.2, 0.3], [0.7, 0.2, 0.9]])
targets = [1, 2]

loss_function = Mean_Squared_Error()

loss = loss_function.forward(outputs, targets)

print(loss)

'''
# Create dataset
X, y = vertical_data(samples=10, classes=3)
# Create model
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
# Create loss function
loss_function = CategoricalCrossEntroy()
# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.bias.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.bias.copy()


start = time.time()
for iteration in range(3000):
    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.bias += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.bias += 0.05 * np.random.randn(1, 3)
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.outputs, y)
    # Calculate accuracy from output of activation2 and targets # calculate values along first axis
    predictions = np.argmax(activation2.outputs, axis=1) 
    accuracy = np.mean(predictions == y)
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        # print('New set of weights found, iteration:', iteration,
        #       'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.bias.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.bias.copy()
        lowest_loss = loss
    # Revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.bias = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.bias = best_dense2_biases.copy()

print(f"best loss = {lowest_loss}")
print(f"time taken = {time.time() - start} seconds")
'''