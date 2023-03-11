import numpy as np

class Layer:
    #parameters are the number of neurons by the number of features
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.5 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros(shape = (1, n_neurons))

    #setting the weights
    def set_weight(self, weight_in):
        self.weights = weight_in

    #setting the biases
    def set_bias(self, bias_in):
        self.bias = bias_in

    #the forward pass of a layer (dot product basically)
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs =  np.dot(inputs, self.weights) + self.bias
    
    #the backward pass of a layer (calculating gradient)
    def backward(self, derivatives):
        self.dweights = np.dot(self.inputs.T, derivatives)
        self.dinputs = np.dot(derivatives, self.weights.T)
        self.dbias = np.sum(derivatives, axis = 0, keepdims = True)

#The relu activation function
class ReLU:
    #the inputs are meant to be the output of a forward pass of a dense layer
    #calculates the ReLU of the inputs
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    #calculates the gradient of the ReLU function with respect to its inputs
    def backward(self, derivatives):
        self.dinputs = derivatives.copy()
        
        self.dinputs[self.inputs <= 0] = 0

#The softmax activation function
class Softmax:
    #the inputs are meant to be the output of a hidden layer typically using a ReLU activation function
    #The forward pass of a softmax function (exponentiation and normalization)
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_inputs / np.sum(exp_inputs, axis = 1, keepdims = True)
        self.outputs = probabilities

    #Derivative of softmax produces Jacobian matrix. (Each output is influenced by all inputs)
    #I will use S to denote the output of the softmax and kd to denote the kronecker delta
    #Derivative = Si,j*kdj,k - Si,j*Si,j
    #i is the sample number
    #j is the output we are taking the derivative of
    #k is the input we are taking the derivative in respect to
    def backward(self, derivatives):
        self.dinputs = np.empty_like(derivatives)
        for i,(single_output, single_dvalue) in enumerate(zip(self.outputs, derivatives)):
            single_output = np.reshape(single_output, (1,-1))

            jacobian = np.diagflat(single_output) - np.dot(single_output.T, single_output)
            
            self.dinputs[i] = np.dot(jacobian, single_dvalue)

#A common loss class 
class Loss:
    #calculates the given loss function and returns its average
    def calculate(self, outputs, y):
        sample_loss = self.forward(outputs, y)
        data_loss = np.mean(sample_loss)
        return data_loss

#Categorical Cross Entropy Loss
class CategoricalCrossEntroy(Loss):
    #The forward pass (-log of the inputs)
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
    
    #The backwards pass (targets / predicted) and then normalized
    def backward(self, outputs, labels):
        samples = len(outputs)

        labels = np.array(labels)
        if (len(labels.shape) == 1):
            labels = np.eye(len(outputs[0]))[labels]
        
        self.dinputs = -labels/outputs 

        self.dinputs /= samples

#The mean squared error class 
class Mean_Squared_Error(Loss):
    #The forward pass, outputs - inputs squared
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

    #The backward pass, calculating the normalized gradient
    def backward(self, outputs, labels):
        self.dinputs = np.copy((2 * outputs - 2 * labels)/len(outputs))

#class to combine both softmax activation functio and categorical cross entropy
class Softmax_Entropy:

    #create both classes to perform forward pass
    def __init__(self):
        self.activation = Softmax()
        self.loss_function = CategoricalCrossEntroy()

    #forward pass, doing activation first and then calculating loss
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)

        self.outputs = self.activation.outputs

        return self.loss_function.calculate(self.outputs, y_true)
    
    #backward pass, derivative simplifies to ytrue - ypred
    def backward(self, outputs, labels):
        labels = np.array(labels)

        samples = len(outputs)
        if (len(labels.shape) == 2): #convert to shortened one hot encoded vector
            labels = np.argmax(labels, axis = 1)
        
        self.dinputs = outputs.copy()

        #subtract 1 if the current label is the correct label for that sample
        self.dinputs[range(samples), labels] -= 1

        #normalize gradient
        self.dinputs /= samples

#the stochastic gradient descent (SGD) optimizer
class SGD:

    #initializes learning rate, learning rate decay, and momentum
    def __init__(self, learning_rate = 1.0, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    
    #the updating of params, adding momentum if needed
    def update_params(self, layer: Layer):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

        if self.momentum:
            if not hasattr(layer, ""):
                layer.weight_updates = np.zeros_like(layer.weights)

                layer.bias_updates = np.zeros_like(layer.bias)
                
            weight_updates = -self.learning_rate * layer.dweights - self.momentum * layer.weight_updates
            bias_updates =  -self.learning_rate * layer.dbias - self.momentum * layer.bias_updates 

            layer.weight_updates = weight_updates
            layer.bias_updates = bias_updates
        
        #if no momentum is added
        else:  
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbias

        layer.weights += weight_updates
        layer.bias += bias_updates

        self. iterations += 1

class Adam:
    def __init__(self, learning_rate=0.01, beta1 = 0.999, beta2 = 0.9, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
    
    def update_params(self, layer: Layer):
        pass

    def post_update_params(self):
        self.iterations += 1
    
