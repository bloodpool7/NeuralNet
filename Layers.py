import numpy as np
import nnfs 
from nnfs.datasets import spiral_data, vertical_data
import time

nnfs.init()

# |A| - matrix A is an input matrix which as the shape (batch amount, number of features)
# | 3 2 1 |
# | 4 2 3 |
# | 4 1 2 |
inputs = [[3, 2, 1, 4, 3],
          [4, 2, 3, 2, 1],
          [4, 1, 2, 3, 1]]
# |B| - matrix B is a weight matrix which has shape (number of neurons, number of inputs)
# | 2 4 1 |
# | 2 1 3 |
# | 3 2 1 |
weights = [[2, 4, 1, 2, 4],
           [2, 1, 3, 3, 2],
           [3, 2, 1, 1, 3],
           [3, 2, 1, 1, 3]]
# we want to do |A| * |B| such that the features are being multiplied by the number of inputs
# therefore, it is sensical to make it so that |B| is actually shaped (number of inputs, number of neurons) 

bias = [3, 2, 1, 4] # bias is a list because there are 3 neurons

# print(np.array(inputs))
# print(np.array(weights).T)

# output = np.dot(inputs, np.array(weights).T) + bias

# print(output)

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
        self.outputs = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.drelu = np.maximum(dvalues, 0)


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

        y_target = np.array(y_target)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if (len(y_target.shape) == 1):
            correct_confidences = y_pred_clipped[range(len(y_pred_clipped)), y_target]
        elif (len(y_target.shape) == 2):
            correct_confidences = np.sum(y_pred_clipped * y_target, axis = 1)

        total_loss = -np.log(correct_confidences)
        return total_loss

class Mean_Squared_Error(Loss):
    def forward(self, y_pred, y_target):
        y_target = np.array(y_target)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1e7)

        if (len(y_target.shape) == 1):
            correct_confidences = y_pred_clipped[range(len(y_pred_clipped)), y_target]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_target, axis = 1)
        
        total_loss = (correct_confidences - 1)**2
        return total_loss


# X, y = spiral_data(samples = 100, classes = 3)

# layer1 = Layer_Dense(2, 10)

# layer2 = Layer_Dense(10, 3)

# activation1 = Activation_ReLU()

# activation2 = Activation_Softmax()

# layer1.forward(X)

# activation1.forward(layer1.outputs)

# layer2.forward(activation1.outputs)

# activation2.forward(layer2.outputs)

# loss_function = CategoricalCrossEntroy()

# loss = loss_function.calculate(activation2.outputs, y)

# predictions = np.argmax(activation2.outputs, axis = 1)

# class_targets = np.argmax(y, axis = 1) if len(y.shape) == 2 else y

# # turns every element of numpy array into 2 decimal points
# print(np.array([list((float("{:.2f}".format(y)) for y in x)) for x in activation2.outputs[:5] ])) 

# print("acc: " + "{:.2f}".format(np.mean(predictions == class_targets)))

# print("loss: " + "{:.2f}".format(loss))

# Create dataset
X, y = vertical_data(samples=1000, classes=3)
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