import numpy as np

class Layer:
    #parameters are the number of neurons by the number of features
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.5 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros(shape = (1, n_neurons))

    #setting the weights
    def set_weight(self, weight_in):
        self.weights = np.array(weight_in)
        self.weights = self.weights.astype(np.float64)

    #setting the biases
    def set_bias(self, bias_in):
        self.bias = np.array(bias_in)
        self.bias = self.bias.astype(np.float64)

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

#an interface for all optimizers
class Optimizer:

    #To be called before parameter updates
    def pre_param_updates(self):
        pass
    
    #To be called to update each layer's weights and biases
    def update_params(self, layer: Layer):
        pass 

    #To be called after all layers' weights and biases have been updated
    def post_param_updates(self):
        pass

#the stochastic gradient descent (SGD) optimizer
class SGD(Optimizer):

    #initializes learning rate, learning rate decay, and momentum
    def __init__(self, learning_rate = 1.0, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
    
    def pre_param_updates(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    #the updating of params, adding momentum if needed
    def update_params(self, layer: Layer):
        if self.momentum:
            if not hasattr(layer, "weight_updates"):
                layer.weight_updates = np.zeros_like(layer.weights)

                layer.bias_updates = np.zeros_like(layer.bias)
                
            weight_updates = -self.current_learning_rate * layer.dweights - self.momentum * layer.weight_updates
            bias_updates =  -self.current_learning_rate * layer.dbias - self.momentum * layer.bias_updates 

            layer.weight_updates = weight_updates
            layer.bias_updates = bias_updates
        
        #if no momentum is added
        else:  
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbias

        layer.weights += weight_updates
        layer.bias += bias_updates

    def post_param_updates(self):
        self. iterations += 1

#The adam optimizer
class Adam(Optimizer):

    #default values for hyperparameters 
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    #To be called per layer
    def update_params(self, layer: Layer):

        #If this is the first iteration, create the necessary attributes for the layer
        if not hasattr(layer, "weight_moments"):
            layer.weight_moments = np.zeros_like(layer.weights)
            layer.weight_velocity = np.zeros_like(layer.weights)

            layer.bias_moments = np.zeros_like(layer.bias)
            layer.bias_velocity = np.zeros_like(layer.bias)

        #Getting all the weight momentums and velocities to make the weight update
        layer.weight_moments = self.beta1 * layer.weight_moments + (1 - self.beta1) * layer.dweights
        layer.weight_velocity = self.beta2 * layer.weight_velocity + (1 - self.beta2) * (layer.dweights ** 2)
        weight_moments_corrected = layer.weight_moments / (1 - self.beta1 ** (self.iterations + 1))
        weight_velocity_corrected = layer.weight_velocity / (1 - self.beta2 ** (self.iterations + 1))

        #Getting all the bias momentums and velocities to make the bias update
        layer.bias_moments = self.beta1 * layer.bias_moments + (1 - self.beta1) * layer.dbias
        layer.bias_velocity = self.beta2 * layer.bias_velocity + (1 - self.beta2) * (layer.dbias ** 2)
        bias_moments_corrected = layer.bias_moments / (1 - self.beta1 ** (self.iterations + 1))
        bias_velocity_corrected = layer.bias_velocity / (1 - self.beta2 ** (self.iterations + 1))

        #updating the params
        layer.weights -= (self.learning_rate * weight_moments_corrected) / (np.sqrt(weight_velocity_corrected) + self.epsilon)
        layer.bias -= (self.learning_rate * bias_moments_corrected) / (np.sqrt(bias_velocity_corrected) + self.epsilon)
            
        
    def post_param_updates(self):
        self.iterations += 1

# A model object
class Model:
    def __init__(self, layers: list = None, activations: list = None, loss = None, optimizer: Optimizer = None):
        self.__layers = layers if layers != None else []
        self.__activations = activations if activations != None else []
        self.__optimizer = optimizer
        self.__loss_function = loss

        if (activations != None and isinstance(self.__activations[-1], Softmax) and isinstance(self.__loss_function, CategoricalCrossEntroy)):
            self.__activations.pop()
            self.__loss_function = Softmax_Entropy()
    
    def __forward(self, inputs, targets):
        '''Forward pass, assumes input comes in 2d array (one batch)
           Will calculate everything but will not return anything. 
           The forward pass should not be called by the user and will
           only be called by other methods in this class. '''
        self.__layers[0].forward(inputs)
        self.__activations[0].forward(self.__layers[0].outputs)

        for i in range(1, len(self.__layers)):
            if (i + 1 == len(self.__layers)):
                if (len(self.__activations) == len(self.__layers)):
                    self.__layers[i].forward(self.__activations[i-1].outputs)
                    self.__activations[i].forward(self.__layers[i].outputs)

                    if (targets != None):
                        self.loss = self.__loss_function.calculate(self.__activations[i].outputs, targets)

                        predictions = np.argmax(self.__activations[-1].outputs, axis=1)
                        if len(targets.shape) == 2:
                            targets = np.argmax(targets, axis=1)
                        self.accuracy = np.mean(predictions == targets) 
                else:
                    self.__layers[i].forward(self.__activations[i-1].outputs)

                    if(targets != None):
                        self.loss = self.__loss_function.forward(self.__layers[i].outputs, targets)

                        predictions = np.argmax(self.__activations[-1].outputs, axis=1)
                        if len(targets.shape) == 2:
                            targets = np.argmax(targets, axis=1)
                        self.accuracy = np.mean(predictions == targets)
            else:

                self.__layers[i].forward(self.__activations[i-1].outputs)
                self.__activations[i].forward(self.__layers[i].outputs)
    
    def __backward(self, labels):
        
        if isinstance(self.__loss_function, Softmax_Entropy):
            self.__loss_function.backward(self.loss_fuction.outputs, labels)
            self.__layers[-1].backward(self.__loss_function.dinputs)

            for i in range(len(self.__activations) - 1, -1 , -1):
                self.__activations[i].backward(self.__layers[i+1].dinputs)
                self.__layers[i].backward(self.__activations[i].dinputs)

        else:
            self.__loss_function.backward(self.__activations[-1].outputs, labels)
            self.__activations[-1].backward(self.__loss_function.dinputs)
            self.__layers[-1].backward(self.__activations[-1].dinputs)

            for i in range(len(self.__layers) - 2, -1, -1):
                self.__activations[i].backward(self.__layers[i + 1].dinputs)
                self.__layers[i].backward(self.__activations[i].dinputs)

        self.__optimizer.pre_param_updates()
        for layer in self.__layers:
            self.__optimizer.update_params(layer)
        self.__optimizer.post_param_updates()

    
    def train(self, X_train = None, X_valid = None, y_train = None, y_valid = None, epochs = None):
        if (len(X_train.shape) == 3):
            for epoch in range(epochs):
                for batch in range(len(X_train)):
                    self.__forward(X_train[batch], y_train[batch])
                    self.__backward(y_train[batch])
        elif (len(X_train.shape) == 2):
            for epoch in range(epochs):
                self.__forward(X_train, y_train)
                self.__backward(y_train)

        self.__forward(X_valid, y_valid)

    def predict(self, X, y = None) -> np.ndarray:
        self.__forward(X, y)

        if (isinstance(self.__loss_function, Softmax_Entropy)):
            return self.__loss_function.outputs
        else:
            return self.__activations[-1].outputs

    #Saves the model using numpy save features
    def save_model(self, weight_name, bias_name):
        weights = []
        biases = []

        for i in range(len(self.__layers)):
            weights.append(self.__layers[i].weights.tolist())
            biases.append(self.__layers[i].bias.tolist())

        np.save(weight_name, np.array(weights, dtype = object), allow_pickle = True)
        np.save(bias_name, np.array(biases, dtype = object), allow_pickle = True)

    #Loads the model using the numpy save features
    def load_model(self, weight_name, bias_name):

        weights = np.load(weight_name, allow_pickle = True)
        biases = np.load(bias_name, allow_pickle = True)

        for arr in weights:
            layer_weights = []
            for lists in arr:
                layer_weights.append(np.array(lists))
            layer_weights = np.array(layer_weights)

            layer = Layer(len(layer_weights), len(layer_weights[0]))
            layer.set_weight(layer_weights)

            self.__layers.append(layer)
        
        for arr in biases:
            layer_biases = []
            for lists in arr:
                layer_biases.append(np.array(lists))
            layer_biases = np.array(layer_biases)
            
            layer.set_bias(layer_biases)
